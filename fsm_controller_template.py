import numpy as np

class CueFSMController:
    """Cue-based: low cue (center) → small turns + full steps (stay); high cue → explore"""
    def __init__(self, max_step=0.1, nd_levels=2):  # nd=2 best per paper
        self.max_step = max_step
        self.nd_levels = nd_levels
        # Map paper's α,ρ to step/turn: low α/high ρ = straight+long → full step/small turn
        self.turn_sigmas = np.array([0.10, 0.60]) if nd_levels == 2 else np.array([0.10, 0.35, 0.70])
        self.step_scales = np.array([1.0, 0.40]) if nd_levels == 2 else np.array([1.0, 0.70, 0.40])

    def step(self, agent_id, obs, rng):
        cue = obs["cue_value"]  # expect 0=center(good), 1=edge(bad)
        nd = min(int(cue * self.nd_levels), self.nd_levels - 1)
        
        step_size = self.step_scales[nd] * self.max_step
        turn_delta = rng.normal(0.0, self.turn_sigmas[nd])
        return {"step_size": step_size, "turn_delta": turn_delta}

class NeighborFSMController:
    """Neighbor-based: many neighbors → cluster (small turns); few → explore (big turns)"""
    def __init__(self, max_step=0.1, nd_levels=3, max_neighbors=20):
        self.max_step = max_step
        self.nd_levels = nd_levels
        self.max_n = max_neighbors
        # Many neigh (nd=0) → cluster like paper's low α/high ρ
        self.turn_sigmas = np.array([0.15, 0.15, 0.50])
        self.step_scales = np.array([0.2, 0.2, 1.0])  # cluster slightly slower

    def step(self, agent_id, obs, rng):
        n_neigh = obs["neighbor_count"]
        norm_n = n_neigh / self.max_n
        nd = min(int((1 - norm_n) * self.nd_levels), self.nd_levels - 1)  # invert: many→low nd
        
        step_size = self.step_scales[nd] * self.max_step
        turn_delta = rng.normal(0.0, self.turn_sigmas[nd])
        return {"step_size": step_size, "turn_delta": turn_delta}
    
class HeteroFSMController:
    def __init__(self, max_step=0.1, cue_ratio=0.4, n_agents=1000, **kwargs):
        self.max_step = max_step
        self.cue_ratio = cue_ratio
        self.cue_controller = CueFSMController(max_step=max_step, **kwargs)
        self.neigh_controller = NeighborFSMController(max_step=max_step, **kwargs)

        # One fixed type per agent for the whole run
        self.is_cue_type = np.random.rand(n_agents) < cue_ratio

    def step(self, agent_id, obs, rng):
        if self.is_cue_type[agent_id]:
            return self.cue_controller.step(agent_id, obs, rng)
        else:
            return self.neigh_controller.step(agent_id, obs, rng)

