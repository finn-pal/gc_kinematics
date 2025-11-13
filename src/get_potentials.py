import argparse
import json
import multiprocessing as mp

import gc_utils
import numpy as np
import pandas as pd

from tools.make_potential import make_potential

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-l", "--location", required=True, type=str, help="either local or katana")
    # parser.add_argument("-c", "--cores", required=False, type=int, help="number of cores to run process on")
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
    parser.add_argument(
        "-p", "--snap_lim", required=False, type=int, help="Minimum snapshot to consider", default=None
    )

    args = parser.parse_args()
    sim = args.simulation
    location = args.location
    snapshots = args.snapshots
    disk_limit = args.disk_limit
    snap_lim = args.snap_lim

    print("Don't forget to add a disk limit")

    if location == "local":
        sim_dir = "../../simulations/"

    elif location == "katana":
        sim_dir = "/srv/scratch/astro/z5114326/simulations/"

    elif location == "one_touch":
        sim_dir = "/Volumes/One_Touch/simulations/"

    elif location == "my_passport":
        sim_dir = "/Volumes/My_Passport_for_Mac/simulations/"

    else:
        raise RuntimeError("Incorrect location provided. Must be local or katana.")

    # sim_dir = "../../simulations/"
    # data_dir = "data/"

    # if snapshots is None:
    #     snap_lst = sim_dir + sim + "/potentials.json"
    #     with open(snap_lst) as snap_json:
    #         snap_data = json.load(snap_json)
    #         snapshots = snap_data[sim]

    public_snapshot_file = sim_dir + "snapshot_times_public.txt"
    pub_data = pd.read_table(public_snapshot_file, comment="#", header=None, sep=r"\s+")
    pub_data.columns = [
        "index",
        "scale_factor",
        "redshift",
        "time_Gyr",
        "lookback_time_Gyr",
        "time_width_Myr",
    ]
    pub_snaps = np.array(pub_data["index"], dtype=int)
    snap_lst = pub_snaps

    if snapshots is None:
        snapshots = np.array(snap_lst)

    if snap_lim is not None:
        snapshots = snapshots[snapshots >= snap_lim]

    # cores = args.cores
    # if cores is None:
    #     # cores = mp.cpu_count()
    #     cores = 1

    print(snapshots)

    halt = gc_utils.get_halo_tree(sim, sim_dir)

    # Create a multiprocessing pool
    # with mp.Pool(processes=cores) as pool:
    #     # Use pool.starmap to run make_potential for each snapshot in parallel
    #     # for snapshot in snapshots:
    #     #     print(snapshot)
    #     pool.starmap(make_potential, [(sim, snapshot, sim_dir, disk_limit) for snapshot in snapshots])

    # the radial extent to which the potential are created needs to have a redshift dependence.

    for snapshot in snapshots:
        make_potential(halt, sim, snapshot, sim_dir, disk_limit)
