package registry

import (
	"errors"
	"sync"
	"time"

	"github.com/gentleman-programming/gentle-mesh/pkg/protocol"
)

var (
	// ErrNodeNotFound is returned when a node is not found in the registry.
	ErrNodeNotFound = errors.New("node not found")
	// ErrNodeAlreadyExists is returned when attempting to register a node that already exists.
	ErrNodeAlreadyExists = errors.New("node already exists")
	// ErrNoNodeAvailable is returned when no healthy node has available capacity.
	ErrNoNodeAvailable = errors.New("no healthy node available with required capacity")
	// ErrAgentNotSupported is returned when no online registered node supports the requested agent.
	ErrAgentNotSupported = errors.New("no registered node supports the requested agent")
	// ErrInvalidNodeInfo is returned when node registration payload is missing required fields.
	ErrInvalidNodeInfo = errors.New("invalid node info: node_id and endpoint must not be empty")
)

// Registry manages federated worker nodes, heartbeats, task routing,
// exclusive branch locks, and idempotency tracking.
type Registry struct {
	mu               sync.RWMutex
	nodes            map[string]*protocol.NodeInfo
	heartbeatTimeout time.Duration
	*BranchLockManager
	*IdempotencyStore
}

// NewRegistry creates a new Registry with the given heartbeat timeout.
// Default branch locking and 24-hour idempotency stores are attached.
func NewRegistry(heartbeatTimeout time.Duration) *Registry {
	return &Registry{
		nodes:             make(map[string]*protocol.NodeInfo),
		heartbeatTimeout:  heartbeatTimeout,
		BranchLockManager: NewBranchLockManager(),
		IdempotencyStore:  NewIdempotencyStore(24 * time.Hour),
	}
}

// Locks returns the exclusive branch lock manager.
func (r *Registry) Locks() *BranchLockManager {
	return r.BranchLockManager
}

// Idempotency returns the idempotency store.
func (r *Registry) Idempotency() *IdempotencyStore {
	return r.IdempotencyStore
}

// RegisterNode registers or updates a node in the mesh registry.
// Validates node_id and endpoint. If the node already exists, updates hardware,
// agents, max_concurrency, endpoint, and sets status to Online with refreshed heartbeat.
func (r *Registry) RegisterNode(req protocol.NodeJoinRequest) (*protocol.NodeInfo, error) {
	if req.NodeID == "" || req.Endpoint == "" {
		return nil, ErrInvalidNodeInfo
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now().Unix()
	existing, exists := r.nodes[req.NodeID]
	if exists {
		existing.Endpoint = req.Endpoint
		existing.Hardware = req.Hardware
		existing.Agents = copyAgents(req.Agents)
		existing.MaxConcurrency = req.MaxConcurrency
		existing.Status = protocol.NodeStatusOnline
		existing.LastHeartbeat = now
		return cloneNodeInfo(existing), nil
	}

	node := &protocol.NodeInfo{
		NodeID:         req.NodeID,
		Endpoint:       req.Endpoint,
		Hardware:       req.Hardware,
		Agents:         copyAgents(req.Agents),
		MaxConcurrency: req.MaxConcurrency,
		ActiveTasks:    0,
		Status:         protocol.NodeStatusOnline,
		LastHeartbeat:  now,
	}
	r.nodes[req.NodeID] = node
	return cloneNodeInfo(node), nil
}

// DeregisterNode removes a node from the registry.
func (r *Registry) DeregisterNode(nodeID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.nodes[nodeID]; !exists {
		return ErrNodeNotFound
	}

	delete(r.nodes, nodeID)
	return nil
}

// Heartbeat processes a periodic keepalive ping from a worker node.
// Updates last heartbeat timestamp, active tasks, and restores status to Online.
func (r *Registry) Heartbeat(req protocol.NodeHeartbeatRequest) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	node, exists := r.nodes[req.NodeID]
	if !exists {
		return ErrNodeNotFound
	}

	timestamp := req.Timestamp
	if timestamp <= 0 {
		timestamp = time.Now().Unix()
	}

	node.LastHeartbeat = timestamp
	node.ActiveTasks = req.ActiveTasks
	node.Status = protocol.NodeStatusOnline
	return nil
}

// GetNode retrieves a point-in-time copy of NodeInfo for the specified node ID.
func (r *Registry) GetNode(nodeID string) (*protocol.NodeInfo, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	node, exists := r.nodes[nodeID]
	if !exists {
		return nil, false
	}
	return cloneNodeInfo(node), true
}

// ListNodes returns a slice of point-in-time copies of all registered nodes.
func (r *Registry) ListNodes() []*protocol.NodeInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]*protocol.NodeInfo, 0, len(r.nodes))
	for _, node := range r.nodes {
		result = append(result, cloneNodeInfo(node))
	}
	return result
}

// PruneStaleNodes checks heartbeat freshness for all nodes:
//   - If time since LastHeartbeat > 3 * heartbeatTimeout, marks node Offline.
//   - Else if time since LastHeartbeat > 1.5 * heartbeatTimeout, marks node Degraded.
func (r *Registry) PruneStaleNodes() {
	if r.heartbeatTimeout <= 0 {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	offlineThreshold := r.heartbeatTimeout * 3
	degradedThreshold := time.Duration(float64(r.heartbeatTimeout) * 1.5)

	for _, node := range r.nodes {
		elapsed := now.Sub(time.Unix(node.LastHeartbeat, 0))
		if elapsed > offlineThreshold {
			node.Status = protocol.NodeStatusOffline
		} else if elapsed > degradedThreshold {
			node.Status = protocol.NodeStatusDegraded
		}
	}
}

// SelectNodeForTask selects the optimal healthy online node to execute a task based on:
//  1. Status and heartbeat freshness: a node whose heartbeats stopped is skipped even
//     while its stored status still says Online, so work is never sent to a dead node.
//  2. Agent and tag capabilities (must support requested agent and all required tags).
//  3. Capacity availability (ActiveTasks < MaxConcurrency; MaxConcurrency <= 0 defaults to 1).
//  4. Load balancing: selects node with lowest ActiveTasks / MaxConcurrency ratio,
//     tie-breaking on lowest ActiveTasks, then deterministic NodeID.
func (r *Registry) SelectNodeForTask(agent string, requiredTags ...string) (*protocol.NodeInfo, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var matchingOnline []*protocol.NodeInfo
	for _, node := range r.nodes {
		if node.Status == protocol.NodeStatusOnline && r.heartbeatIsFresh(node) && nodeSupportsAgent(node, agent, requiredTags) {
			matchingOnline = append(matchingOnline, node)
		}
	}

	if len(matchingOnline) == 0 {
		return nil, ErrAgentNotSupported
	}

	var available []*protocol.NodeInfo
	for _, node := range matchingOnline {
		effectiveMax := node.MaxConcurrency
		if effectiveMax <= 0 {
			effectiveMax = 1
		}
		if node.ActiveTasks < effectiveMax {
			available = append(available, node)
		}
	}

	if len(available) == 0 {
		return nil, ErrNoNodeAvailable
	}

	var best *protocol.NodeInfo
	var bestRatio float64

	for _, node := range available {
		effectiveMax := node.MaxConcurrency
		if effectiveMax <= 0 {
			effectiveMax = 1
		}
		ratio := float64(node.ActiveTasks) / float64(effectiveMax)

		if best == nil {
			best = node
			bestRatio = ratio
			continue
		}

		if ratio < bestRatio {
			best = node
			bestRatio = ratio
		} else if ratio == bestRatio {
			if node.ActiveTasks < best.ActiveTasks {
				best = node
				bestRatio = ratio
			} else if node.ActiveTasks == best.ActiveTasks && node.NodeID < best.NodeID {
				best = node
				bestRatio = ratio
			}
		}
	}

	return cloneNodeInfo(best), nil
}

// heartbeatIsFresh reports whether the node's last heartbeat is recent enough to keep
// sending it work. It applies the same 1.5x heartbeatTimeout that PruneStaleNodes uses
// to mark a node Degraded, but on read: a node that stopped beating stops being selected
// immediately, without waiting for the pruning ticker to run.
func (r *Registry) heartbeatIsFresh(node *protocol.NodeInfo) bool {
	if r.heartbeatTimeout <= 0 {
		return true
	}
	return time.Since(time.Unix(node.LastHeartbeat, 0)) <= time.Duration(float64(r.heartbeatTimeout)*1.5)
}

// MarkNodeUnreachable marks a node Degraded after a dispatch failed at the transport
// level, so it leaves the rotation right away instead of waiting for the heartbeat
// timeout. It is a soft mark: the next heartbeat restores the node to Online, so one
// failed dispatch cannot retire an otherwise healthy node.
func (r *Registry) MarkNodeUnreachable(nodeID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	node, exists := r.nodes[nodeID]
	if !exists {
		return ErrNodeNotFound
	}
	node.Status = protocol.NodeStatusDegraded
	return nil
}

func nodeSupportsAgent(node *protocol.NodeInfo, agent string, requiredTags []string) bool {
	for _, profile := range node.Agents {
		if profile.Name == agent {
			if hasAllTags(profile.Tags, requiredTags) {
				return true
			}
		}
	}
	return false
}

func hasAllTags(profileTags []string, requiredTags []string) bool {
	if len(requiredTags) == 0 {
		return true
	}
	tagSet := make(map[string]struct{}, len(profileTags))
	for _, t := range profileTags {
		tagSet[t] = struct{}{}
	}
	for _, req := range requiredTags {
		if _, ok := tagSet[req]; !ok {
			return false
		}
	}
	return true
}

func copyAgents(agents []protocol.AgentProfile) []protocol.AgentProfile {
	if agents == nil {
		return nil
	}
	cp := make([]protocol.AgentProfile, len(agents))
	for i, a := range agents {
		cp[i] = a
		if a.Tags != nil {
			cp[i].Tags = make([]string, len(a.Tags))
			copy(cp[i].Tags, a.Tags)
		}
	}
	return cp
}

func cloneNodeInfo(node *protocol.NodeInfo) *protocol.NodeInfo {
	if node == nil {
		return nil
	}
	cp := *node
	cp.Agents = copyAgents(node.Agents)
	return &cp
}
