import argparse
import json
import multiprocessing as mp

import gizmo_analysis as gizmo
import numpy as np
import utilities as ut
from gc_utils import (  # noqa: F401
    block_print,
    enable_print,
    get_halo_tree,
    main_prog_halt,
    snapshot_name,
)
from scipy.optimize import minimize_scalar
from tqdm import tqdm


def mass_dif(r_test, part, host_name, gas_tot):
    gas_mask_test = part["gas"].prop(f"{host_name}.distance.principal.total") <= r_test
    gas_mass_test = np.sum(part["gas"]["mass"][gas_mask_test])

    mass_dif = np.abs((gas_tot / 2) - gas_mass_test)
    return mass_dif


def get_r50(host_index, halt, part):
    r_vir = halt["radius"][host_index]
    host_name = ut.catalog.get_host_name(0)

    gas_mask = part["gas"].prop(f"{host_name}.distance.principal.total") < r_vir
    gas_tot = np.sum(part["gas"]["mass"][gas_mask])

    res = minimize_scalar(mass_dif, bounds=(0, r_vir), method="bounded", args=(part, host_name, gas_tot))
    r_50 = res.x

    return r_50


def kappa_co(part, snapshot: int, r_50: float, kappa_dict: dict = {}, log_t_max: float = 4.5):
    host_name = ut.catalog.get_host_name(0)
    # get star values ####################################
    # only select stars within r_50
    star_mask = part["star"].prop(f"{host_name}.distance.principal.total") < r_50

    # get 3D positions and velocities
    vel_xyz_star = part["star"].prop(f"{host_name}.velocity.principal")[star_mask]
    pos_xyz_star = part["star"].prop(f"{host_name}.distance.principal")[star_mask]

    # particle distances from z-axis
    star_rho = part["star"].prop(f"{host_name}.distance.principal.cylindrical")[:, 0][star_mask]

    # star mass
    star_mass = part["star"]["mass"][star_mask]

    # total stellar kinematic energy
    ek_tot_star = (star_mass) * 0.5 * np.linalg.norm(vel_xyz_star, axis=1) ** 2

    # get angular momentume and create mask
    lz_star = (pos_xyz_star[:, 0] * vel_xyz_star[:, 1] - pos_xyz_star[:, 1] * vel_xyz_star[:, 0]) * star_mass
    lz_star_mask = lz_star > 0

    # get energies of co-rotating particles
    K_co_star = np.sum(
        0.5
        * star_mass[lz_star_mask]
        * (lz_star[lz_star_mask] / (star_mass[lz_star_mask] * star_rho[lz_star_mask])) ** 2
    )

    # get gas values #####################################
    # only select cold gas particles within r_50
    tsel = np.log10(part["gas"]["temperature"]) < log_t_max
    gas_mask = part["gas"].prop(f"{host_name}.distance.principal.total") < r_50

    # get 3D positions and velocities
    vel_xyz_gas = part["gas"].prop(f"{host_name}.velocity.principal")[gas_mask & tsel]
    pos_xyz_gas = part["gas"].prop(f"{host_name}.distance.principal")[gas_mask & tsel]

    # particle distances from z-axis
    gas_rho = part["gas"].prop(f"{host_name}.distance.principal.cylindrical")[:, 0][gas_mask & tsel]

    # gas mass
    gas_mass = part["gas"]["mass"][gas_mask & tsel]

    # total gas kinematic energy
    ek_tot_gas = (gas_mass) * 0.5 * np.linalg.norm(vel_xyz_gas, axis=1) ** 2

    # combined total energy
    ek_tot = np.sum(ek_tot_star) + np.sum(ek_tot_gas)

    # get angular momentume and create mask
    lz_gas = (pos_xyz_gas[:, 0] * vel_xyz_gas[:, 1] - pos_xyz_gas[:, 1] * vel_xyz_gas[:, 0]) * gas_mass
    lz_gas_mask = lz_gas > 0

    # get energies of co-rotating particles
    K_co_gas = np.sum(
        0.5
        * gas_mass[lz_gas_mask]
        * (lz_gas[lz_gas_mask] / (gas_mass[lz_gas_mask] * gas_rho[lz_gas_mask])) ** 2
    )

    # combine stars and cold gas ###################################
    ek_tot = np.sum(ek_tot_star) + np.sum(ek_tot_gas)
    K_co = K_co_star + K_co_gas

    kappa_co = K_co / ek_tot

    snap_id = snapshot_name(snapshot)
    kappa_dict[snap_id] = kappa_co

    return kappa_dict


def call_func(idx, halt, fire_dir: str, kappa_dict: dict):
    snapshot = halt["snapshot"][idx]
    ptypes = ["star", "gas"]
    for _ in tqdm(range(1), ncols=150, desc="Retrieving Snapshot %d.................." % snapshot):
        block_print()  # block verbose print statements
        part = gizmo.io.Read.read_snapshots(ptypes, "index", snapshot, fire_dir, assign_hosts_rotation=True)
        enable_print()

    r_50 = get_r50(idx, halt, part)
    kappa_dict = kappa_co(part, snapshot, r_50, kappa_dict)
    del part

    return kappa_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
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
    sim = args.simulation

    # directories
    sim_dir = "../../simulations/"
    data_dir = "data/"
    sim_codes = data_dir + "external/simulation_codes.json"
    fire_dir = sim_dir + sim + "/" + sim + "_res7100/"

    potential_snaps = data_dir + "external/potentials.json"
    with open(potential_snaps) as json_file:
        pot_data = json.load(json_file)

    with open(sim_codes) as json_file:
        sim_data = json.load(json_file)

    snap_lst = args.snapshots
    if snap_lst is None:
        snap_lst = np.array(pot_data[sim], dtype=int)

    main_halo_tid = [sim_data[sim]["halo"]]

    halt = get_halo_tree(sim, sim_dir)
    tid_main_lst = main_prog_halt(halt, main_halo_tid)

    idx_main_lst = []
    for tid in tid_main_lst:
        idx = np.where(halt["tid"] == tid)[0][0]
        if halt["snapshot"][idx] in snap_lst:
            idx_main_lst.append(idx)

    cores = args.cores
    if cores is None:
        cores = min(5, len(idx_main_lst))  # Limit cores to task size

    with mp.Manager() as manager:
        shared_dict = manager.dict()  # Shared dictionary across processes

        # Construct args inside the Manager context
        args = [(idx, halt, fire_dir, shared_dict) for idx in idx_main_lst]

        with mp.Pool(processes=cores, maxtasksperchild=1) as pool:
            pool.starmap(call_func, args, chunksize=1)

        result_dict = dict(shared_dict)

    save_path = data_dir + "external/kappa_co.json"
    with open(save_path, "w") as json_file:
        json.dump(result_dict, json_file, indent=4)

    del halt
