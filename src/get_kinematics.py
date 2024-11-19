import argparse
import json
import multiprocessing as mp

import h5py
import numpy as np
from gc_utils import iteration_name, snapshot_name  # type: ignore

from tools.gc_kinematics import get_kinematics


def add_kinematics_hdf5(simulation, it_lst: list[int], snap_lst: list[int], result_dict: dict, sim_dir: str):
    proc_file = sim_dir + simulation + "/" + simulation + "_processed.hdf5"
    proc_data = h5py.File(proc_file, "a")  # open processed data file

    for it in it_lst:
        it_id = iteration_name(it)
        if it_id in proc_data.keys():
            it_grouping = proc_data[it_id]
        else:
            it_grouping = proc_data.create_group(it_id)
        if "snapshots" in it_grouping.keys():
            snap_groups = it_grouping["snapshots"]
        else:
            snap_groups = it_grouping.create_group("snapshots")
        for snap in snap_lst:
            snap_id = snapshot_name(snap)
            if snap_id in snap_groups.keys():
                snapshot = snap_groups[snap_id]
            else:
                snapshot = snap_groups.create_group(snap_id)
            for key in result_dict[snap_id][it_id].keys():
                if key in snapshot.keys():
                    del snapshot[key]
                else:
                    snapshot.create_dataset(key, data=result_dict[snap_id][it_id][key])

    proc_data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-a", "--iteration_low_limit", required=True, type=int, help="lower bound iteration")
    parser.add_argument("-b", "--iteration_up_limit", required=True, type=int, help="upper bound iteration")
    parser.add_argument("-c", "--cores", required=False, type=int, help="number of cores to run process on")
    parser.add_argument(
        "-n",
        "--snapshots",
        required=False,
        nargs="+",
        type=int,
        help="list of snapshots of interest, if None provided will default to all publicly available",
    )

    args = parser.parse_args()

    it_min = args.iteration_low_limit
    it_max = args.iteration_up_limit
    it_lst = np.linspace(it_min, it_max, it_max - it_min + 1, dtype=int)

    sim = args.simulation

    sim_dir = "../../simulations/"
    data_dir = "data/"

    potential_snaps = data_dir + "external/potentials.json"
    with open(potential_snaps) as json_file:
        pot_data = json.load(json_file)

    snap_lst = args.snapshots
    if snap_lst is None:
        snap_lst = np.array(pot_data[sim], dtype=int)

    cores = args.cores
    if cores is None:
        # 4 cores is max to run with 64 GB RAM
        cores = mp.cpu_count()

    # look into this such that evenly sampled across time. as later snapshots are much faster
    snap_groups = np.array_split(snap_lst, cores)

    with mp.Manager() as manager:
        shared_dict = manager.dict()  # Shared dictionary across processes
        args = [(sim, it_lst, snap, sim_dir, data_dir, shared_dict) for snap in snap_lst]

        with mp.Pool(processes=cores, maxtasksperchild=1) as pool:
            pool.starmap(get_kinematics, args, chunksize=1)

        result_dict = dict(shared_dict)

    add_kinematics_hdf5(sim, it_lst, snap_lst, result_dict, sim_dir)
