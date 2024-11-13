import argparse
import json

from tools.make_potential import make_potential
from tqdm import tqdm


def main(simulation: str, snapshots: list[int] = None):
    sim_dir = "/Users/z5114326/Documents/simulations/"
    data_dir = "/Users/z5114326/Documents/GitHub/gc_kinematics_new/data/"

    if snapshots is None:
        snap_lst = data_dir + "external/potentials.json"
        with open(snap_lst) as snap_json:
            snap_data = json.load(snap_json)
            snapshots = snap_data[simulation]

    for snapshot in tqdm(
        snapshots, ncols=150, total=len(snapshots), desc="Retrieving Potentials for Snapshots............."
    ):
        make_potential(simulation, snapshot, sim_dir, data_dir)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument(
        "-n",
        "--snapshots",
        required=False,
        nargs="+",
        type=int,
        help="list of snapshots of interest, if None provided will default to all publicly available",
    )

    args = parser.parse_args()

    main(args.simulation, args.snapshots)
