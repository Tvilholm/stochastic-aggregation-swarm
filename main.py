from simulator import SwarmSimulator
from fsm_controller_template import SimpleFSMController
import numpy as np
import matplotlib.pyplot as plt

def cue_field(x, y):
    # Example: high cue in the top-right corner
    return np.exp(-((x - 8) ** 2 + (y - 8) ** 2) / 4.0)

if __name__ == "__main__":
    controller = SimpleFSMController(max_step=0.1)

    sim = SwarmSimulator(
        n_agents=200,
        arena_size=(10.0, 10.0),
        dt=0.1,
        max_step=0.1,
        neighbor_radius=0.5,
        cue_field=cue_field,
        controller=controller,
        rng_seed=0,
    )

    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))

    for t in range(5000):
        sim.step()
        pos = sim.get_positions()
        ax.clear()
        ax.set_xlim(0, sim.W)
        ax.set_ylim(0, sim.H)
        ax.scatter(pos[:, 0], pos[:, 1], s=10)
        ax.set_title(f"t={t}")
        plt.pause(0.01)

    plt.ioff()
    plt.show()
