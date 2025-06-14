import argparse
import json
import multiprocessing as mp

from tools.make_potential import make_potential

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-c", "--cores", required=False, type=int, help="number of cores to run process on")
    parser.add_argument(
        "-d", "--disk_limit", default=None, required=False, type=int, help="snapshot when disk formed"
    )
    parser.add_argument(
        "-n",
        "--snapshots",
        required=False,
        nargs="+",
        type=int,
        help="list of snapshots of interest, if None provided will default to all publicly available",
    )

    args = parser.parse_args()
    sim = args.simulation
    snapshots = args.snapshots
    disk_limit = args.disk_limit

    print("Don't forget to add a disk limit")

    sim_dir = "../../simulations/"
    # data_dir = "data/"

    if snapshots is None:
        snap_lst = sim_dir + sim + "/potentials.json"
        with open(snap_lst) as snap_json:
            snap_data = json.load(snap_json)
            snapshots = snap_data[sim]

    cores = args.cores
    if cores is None:
        # cores = mp.cpu_count()
        cores = 1

    # Create a multiprocessing pool
    with mp.Pool(processes=cores) as pool:
        # Use pool.starmap to run make_potential for each snapshot in parallel
        # for snapshot in snapshots:
        #     print(snapshot)
        pool.starmap(make_potential, [(sim, snapshot, sim_dir, disk_limit) for snapshot in snapshots])

        # the radial extent to which the potential are created needs to have a redshift dependence.
