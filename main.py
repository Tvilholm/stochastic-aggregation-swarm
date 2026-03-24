import argparse
from simulator import SwarmSimulator
from fsm_controller_template import CueFSMController, NeighborFSMController, HeteroFSMController
import numpy as np
import matplotlib.pyplot as plt

def cue_field(x, y):
    return np.exp(-((x - 5)**2 + (y - 5)**2) / 4.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--controller", type=str, default="hetero",
                        choices=["cue", "neighbor", "hetero"])
    parser.add_argument("--n_agents", type=int, default=200)

    args = parser.parse_args()

    # Select controller
    if args.controller == "cue":
        controller = CueFSMController()
    elif args.controller == "neighbor":
        controller = NeighborFSMController()
    else:
        controller = HeteroFSMController()

    sim = SwarmSimulator(
        n_agents=args.n_agents,
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

    for t in range(3600):
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