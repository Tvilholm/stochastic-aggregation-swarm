import numpy as np

class SimpleFSMController:
    """
    Example FSM controller:
    - States: 0 = explore, 1 = cluster
    - Input: quantized neighbor count, quantized cue value
    """

    def __init__(
        self,
        neighbor_bins=(0, 3, 6, 999),
        cue_bins=(0.0, 0.3, 0.7, 1e6),
        max_step=0.1,
    ):
        self.max_step = max_step
        self.neighbor_bins = neighbor_bins
        self.cue_bins = cue_bins

        # Per-agent internal states
        self.states = {}  # agent_id -> int state

    # ------------- public entry point ------------- #
    def step(self, agent_id, obs, rng):
        # Ensure agent has a state
        if agent_id not in self.states:
            self.states[agent_id] = 0  # start in "explore"

        state = self.states[agent_id]

        # Read local observations
        n = obs["neighbor_count"]
        c = obs["cue_value"]

        n_q = self._quantize_neighbors(n)
        c_q = self._quantize_cue(c)

        # FSM transition
        new_state = self._transition(state, n_q, c_q, rng)
        self.states[agent_id] = new_state

        # Output motion parameters based on new_state
        step_size, dtheta = self._output(new_state, rng)
        return {"step_size": step_size, "turn_delta": dtheta}

    # ------------- helper methods ------------- #
    def _quantize_neighbors(self, n):
        # returns bin index 0,1,2,...
        for i in range(len(self.neighbor_bins) - 1):
            if self.neighbor_bins[i] <= n < self.neighbor_bins[i + 1]:
                return i
        return len(self.neighbor_bins) - 2

    def _quantize_cue(self, c):
        for i in range(len(self.cue_bins) - 1):
            if self.cue_bins[i] <= c < self.cue_bins[i + 1]:
                return i
        return len(self.cue_bins) - 2

    def _transition(self, state, n_q, c_q, rng):
        """
        Example rules:
        - if few neighbors and low cue -> explore
        - if many neighbors or high cue -> cluster
        """
        if state == 0:  # explore
            if n_q >= 2 or c_q >= 2:
                # with some probability, switch to cluster
                if rng.random() < 0.7:
                    return 1
            return 0
        elif state == 1:  # cluster
            if n_q == 0 and c_q == 0:
                # leave cluster if isolated and in low cue
                if rng.random() < 0.5:
                    return 0
            return 1
        else:
            return 0

    def _output(self, state, rng):
        if state == 0:  # explore: larger turns, larger steps
            step = self.max_step
            dtheta = rng.normal(0.0, 0.6)
        elif state == 1:  # cluster: smaller steps, smaller turns
            step = 0.3 * self.max_step
            dtheta = rng.normal(0.0, 0.1)
        else:
            step = self.max_step
            dtheta = rng.normal(0.0, 0.6)
        return step, dtheta
