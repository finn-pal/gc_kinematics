import argparse
import json

import numpy as np
from gc_utils import open_snapshot  # type: ignore
from tqdm import tqdm

from tools.gc_kinematics import get_kinematics


def main(simulation: str, it_min: int, it_max: int):
    sim_dir = "/Users/z5114326/Documents/simulations/"
    data_dir = "/Users/z5114326/Documents/GitHub/gc_kinematics/data/"

    fire_dir = sim_dir + simulation + "/" + simulation + "_res7100/"

    snap_lst = np.array([600], dtype=int)

    potential_snaps = data_dir + "external/potentials.json"
    with open(potential_snaps) as json_file:
        pot_data = json.load(json_file)

    # snap_lst = np.array(pot_data[simulation], dtype=int)
    # snap_lst = snap_lst[2:]

    it_lst = np.linspace(it_min, it_max, it_max - it_min + 1, dtype=int)

    for snapshot in tqdm(
        snap_lst, total=len(snap_lst), ncols=150, desc="Retrieving Kinematics...................."
    ):
        part = open_snapshot(snapshot, fire_dir)
        get_kinematics(part, simulation, it_lst, snapshot, sim_dir, data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-l", "--it_min", required=True, type=int, help="lower bound iteration")
    parser.add_argument("-u", "--it_max", required=True, type=int, help="upper bound iteration")
    args = parser.parse_args()

    main(args.simulation, args.it_min, args.it_max)
