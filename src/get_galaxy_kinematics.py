import argparse
import json
import multiprocessing as mp
import os
import time

import gc_utils
import halo_analysis as halo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utilities as ut
from matplotlib import colors

_global_halt = None


def init_worker(halt_obj):
    global _global_halt
    _global_halt = halt_obj


def constant_star_snap(halt, main_halo_tid: int, sim_dir: str = None, get_public_snap: bool = True):
    idx = np.where(halt["tid"] == main_halo_tid)[0][0]

    while halt["star.radius.90"][idx] > 0:
        star_snap = halt["snapshot"][idx]
        idx = halt["progenitor.main.index"][idx]

        snap_lst = np.arange(star_snap, 601)

    if get_public_snap:
        if sim_dir is None:
            raise RuntimeError("Need to provide sim_dir if get_public_snap is True")

        snap_pub_dir = sim_dir + "/snapshot_times_public.txt"
        snap_pub_data = pd.read_table(snap_pub_dir, comment="#", header=None, sep=r"\s+")
        snap_pub_data.columns = [
            "index",
            "scale_factor",
            "redshift",
            "time_Gyr",
            "lookback_time_Gyr",
            "time_width_Myr",
        ]
        snap_lst = snap_pub_data["index"]

        snap_lst = np.array(snap_lst[snap_lst >= star_snap])

    return snap_lst


def get_halo_center(part, halt, main_halo_tid, sim, sim_dir, snapshot):
    not_host_snap_lst = gc_utils.get_different_snap_lst(main_halo_tid, halt, sim, sim_dir)

    # is the MW progenitor is the main host at this snapshot
    is_main_host = snapshot not in not_host_snap_lst

    # check if centering should be on dm halo
    sim_codes = sim_dir + "simulation_codes.json"
    with open(sim_codes) as sim_json:
        sim_data = json.load(sim_json)

    halt_center_snap_lst = sim_data[sim]["dm_center"]
    use_dm_center = snapshot in halt_center_snap_lst

    # if the halo is not the host at this snapshot or it has been flagged to use dm center at this snapshot
    if (not is_main_host) or (use_dm_center):
        # get MW progenitor halo details at this snapshot
        halo_tid = gc_utils.get_main_prog_at_snap(halt, main_halo_tid, snapshot)

        if use_dm_center:
            halo_detail_dict = gc_utils.get_dm_halo_details(part, halt, halo_tid, snapshot, True)
        else:
            halo_detail_dict = gc_utils.get_halo_details(part, halt, halo_tid, snapshot)

        return_dict = {"use_host_prop": False, "halo_details": halo_detail_dict}

    else:
        return_dict = {"use_host_prop": True}

    return return_dict


def get_kappa_co(
    halt,
    part,
    main_halo_tid: int,
    sim: str,
    sim_dir: str,
    snapshot: int,
    r_limit: float,
    disk_ptypes: list[str] = ["star", "gas"],
    log_t_max: float = 4.5,
):
    # Combination of Correa 2017, Thob 2019, Jiminez 2023

    host_return_dict = get_halo_center(part, halt, main_halo_tid, sim, sim_dir, snapshot)

    # create dict
    kappa_co_dict = {}

    if "star" in disk_ptypes:
        if host_return_dict["use_host_prop"]:
            # only select stars within r_limit
            star_mask = part["star"].prop("host.distance.principal.total") < r_limit

            # get 3D positions and velocities
            vel_xyz_star = part["star"].prop("host.velocity.principal")[star_mask]
            pos_xyz_star = part["star"].prop("host.distance.principal")[star_mask]

            # particle distances from z-axis
            star_rho = part["star"].prop("host.distance.principal.cylindrical")[:, 0][star_mask]

            # star mass
            star_mass = part["star"]["mass"][star_mask]

        else:
            halo_detail_dict = host_return_dict["halo_details"]

            # only select stars within r_limit
            star_pos_mask = (
                ut.particle.get_distances_wrt_center(
                    part,
                    species=["star"],
                    center_position=halo_detail_dict["position"],
                    rotation=halo_detail_dict["rotation"],
                    coordinate_system="cartesian",
                    total_distance=True,
                )
                < r_limit
            )

            # get 3D positions and velocities
            vel_xyz_star = ut.particle.get_velocities_wrt_center(
                part,
                species=["star"],
                center_position=halo_detail_dict["position"],
                center_velocity=halo_detail_dict["velocity"],
                rotation=halo_detail_dict["rotation"],
                coordinate_system="cartesian",
                total_velocity=False,
            )[star_pos_mask]

            pos_xyz_star = ut.particle.get_distances_wrt_center(
                part,
                species=["star"],
                center_position=halo_detail_dict["position"],
                rotation=halo_detail_dict["rotation"],
                coordinate_system="cartesian",
                total_distance=False,
            )[star_pos_mask]

            # particle distances from z-axis
            star_rho = ut.particle.get_distances_wrt_center(
                part,
                species=["star"],
                center_position=halo_detail_dict["position"],
                rotation=halo_detail_dict["rotation"],
                coordinate_system="cylindrical",
                total_distance=False,
            )[:, 0][star_pos_mask]

            # star mass
            star_mass = part["star"]["mass"][star_pos_mask]

        # total stellar kinematic energy within r_limit (don't include lz>0 mask)
        ek_tot_s = 0.5 * np.sum(star_mass * np.linalg.norm(vel_xyz_star, axis=1) ** 2)

        # get angular momentume and create mask
        lz_star = (
            pos_xyz_star[:, 0] * vel_xyz_star[:, 1] - pos_xyz_star[:, 1] * vel_xyz_star[:, 0]
        ) * star_mass
        lz_star_mask = lz_star > 0

        k_rot_s = np.sum(
            0.5
            * star_mass[lz_star_mask]
            * (lz_star[lz_star_mask] / (star_mass[lz_star_mask] * star_rho[lz_star_mask])) ** 2
        )

        # get energies of co-rotating particles
        kappa_co_s = k_rot_s / ek_tot_s

        kappa_co_dict["kappa_co_s"] = kappa_co_s

    if "gas" in disk_ptypes:
        # only select gas particles within r_limit
        gas_tem_mask = np.log10(part["gas"]["temperature"]) < log_t_max

        if host_return_dict["use_host_prop"]:
            # only select gas within r_limit
            gas_pos_mask = part["gas"].prop("host.distance.principal.total") < r_limit

            # get 3D positions and velocities
            vel_xyz_gas = part["gas"].prop("host.velocity.principal")[gas_pos_mask & gas_tem_mask]
            pos_xyz_gas = part["gas"].prop("host.distance.principal")[gas_pos_mask & gas_tem_mask]

            # particle distances from z-axis
            gas_rho = part["gas"].prop("host.distance.principal.cylindrical")[:, 0][
                gas_pos_mask & gas_tem_mask
            ]

            # gas mass
            gas_mass = part["gas"]["mass"][gas_pos_mask & gas_tem_mask]

        else:
            halo_detail_dict = host_return_dict["halo_details"]

            # only select gas within r_limit
            gas_pos_mask = (
                ut.particle.get_distances_wrt_center(
                    part,
                    species=["gas"],
                    center_position=halo_detail_dict["position"],
                    rotation=halo_detail_dict["rotation"],
                    coordinate_system="cartesian",
                    total_distance=True,
                )
                < r_limit
            )

            # get 3D positions and velocities
            vel_xyz_gas = ut.particle.get_velocities_wrt_center(
                part,
                species=["gas"],
                center_position=halo_detail_dict["position"],
                center_velocity=halo_detail_dict["velocity"],
                rotation=halo_detail_dict["rotation"],
                coordinate_system="cartesian",
                total_velocity=False,
            )[gas_pos_mask & gas_tem_mask]

            pos_xyz_gas = ut.particle.get_distances_wrt_center(
                part,
                species=["gas"],
                center_position=halo_detail_dict["position"],
                rotation=halo_detail_dict["rotation"],
                coordinate_system="cartesian",
                total_distance=False,
            )[gas_pos_mask & gas_tem_mask]

            # particle distances from z-axis
            gas_rho = ut.particle.get_distances_wrt_center(
                part,
                species=["gas"],
                center_position=halo_detail_dict["position"],
                rotation=halo_detail_dict["rotation"],
                coordinate_system="cylindrical",
                total_distance=False,
            )[:, 0][gas_pos_mask & gas_tem_mask]

            # gas mass
            gas_mass = part["gas"]["mass"][gas_pos_mask & gas_tem_mask]

        # total gas kinematic energy within r_limit (don't include lz>0 mask)
        ek_tot_g = 0.5 * np.sum(gas_mass * np.linalg.norm(vel_xyz_gas, axis=1) ** 2)

        # get angular momentume and create mask
        lz_gas = (pos_xyz_gas[:, 0] * vel_xyz_gas[:, 1] - pos_xyz_gas[:, 1] * vel_xyz_gas[:, 0]) * gas_mass
        lz_gas_mask = lz_gas > 0

        k_rot_g = np.sum(
            0.5
            * gas_mass[lz_gas_mask]
            * (lz_gas[lz_gas_mask] / (gas_mass[lz_gas_mask] * gas_rho[lz_gas_mask])) ** 2
        )

        # get energies of co-rotating particles
        kappa_co_g = k_rot_g / ek_tot_g

        kappa_co_dict["kappa_co_g"] = kappa_co_g

    if ("star" in disk_ptypes) and ("gas" in disk_ptypes):
        kappa_co_sg = (k_rot_s + k_rot_g) / (ek_tot_s + ek_tot_g)
        kappa_co_dict["kappa_co_sg"] = kappa_co_sg

    return kappa_co_dict


def get_v_sigma(
    halt,
    part,
    main_halo_tid: int,
    sim: str,
    sim_dir: str,
    snapshot: int,
    r_limit: float,
    disk_ptypes: list[str] = ["star", "gas"],
    log_t_max: float = 4.5,
    bin_size: float = 0.2,  # kpc
    bin_num: int = None,
):
    # We define Vrot as the maximum of the rotation curve
    # Sigma as the as the median of the velocity dispersion profile
    # Look at Kareen El-Badry 2018 Gas Kinematics, morphology

    host_return_dict = get_halo_center(part, halt, main_halo_tid, sim, sim_dir, snapshot)
    if (bin_size is not None) and (bin_num is not None):
        raise RuntimeError("Only select one method for bin creation")

    if (bin_size is None) and (bin_num is None):
        raise RuntimeError("Need to select a method for bin creation")

    if bin_size is not None:
        bin_edges = np.arange(0, r_limit + bin_size, bin_size)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_num = len(bin_centers)

    if bin_num is not None:
        bin_edges = np.linspace(0, r_limit, bin_num + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # create dict
    v_sig_dict = {}

    # stellar information
    if "star" in disk_ptypes:
        if host_return_dict["use_host_prop"]:
            # only select stars within r_limit
            star_pos_mask = part["star"].prop("host.distance.principal.total") < r_limit

            star_cyl_rad = part["star"].prop("host.distance.principal.cylindrical")[:, 0][star_pos_mask]
            star_rot_vel = part["star"].prop("host.velocity.principal.cylindrical")[:, 1][star_pos_mask]

        else:
            halo_detail_dict = host_return_dict["halo_details"]

            # only select stars within r_limit
            star_pos_mask = (
                ut.particle.get_distances_wrt_center(
                    part,
                    species=["star"],
                    center_position=halo_detail_dict["position"],
                    rotation=halo_detail_dict["rotation"],
                    coordinate_system="cartesian",
                    total_distance=True,
                )
                < r_limit
            )

            star_cyl_rad = ut.particle.get_distances_wrt_center(
                part,
                species=["star"],
                center_position=halo_detail_dict["position"],
                rotation=halo_detail_dict["rotation"],
                coordinate_system="cylindrical",
                total_distance=False,
            )[:, 0][star_pos_mask]

            star_rot_vel = ut.particle.get_velocities_wrt_center(
                part,
                species=["star"],
                center_position=halo_detail_dict["position"],
                center_velocity=halo_detail_dict["velocity"],
                rotation=halo_detail_dict["rotation"],
                coordinate_system="cylindrical",
                total_velocity=False,
            )[:, 1][star_pos_mask]

        star_mass = part["star"]["mass"][star_pos_mask]

        v_rot_star_arr = np.zeros(bin_num)
        sig_rot_star_arr = np.zeros(bin_num)
        for i in range(bin_num):
            in_bin = (star_cyl_rad >= bin_edges[i]) & (star_cyl_rad < bin_edges[i + 1])
            if np.any(in_bin):
                weights = star_mass[in_bin]
                v_rot_star_i = np.average(star_rot_vel[in_bin], weights=weights)
                v_rot_star_i_2 = np.average(star_rot_vel[in_bin] ** 2, weights=weights)

                sig_rot_star_i = np.sqrt(v_rot_star_i_2 - v_rot_star_i**2)

                v_rot_star_arr[i] = v_rot_star_i
                sig_rot_star_arr[i] = sig_rot_star_i

            else:
                v_rot_star_arr[i] = np.nan
                sig_rot_star_arr[i] = np.nan

        v_rot_s = np.nanmax(v_rot_star_arr)
        sig_rot_s = np.nanmedian(sig_rot_star_arr)

        v_sig_dict["v_rot_s"] = v_rot_s
        v_sig_dict["sig_rot_s"] = sig_rot_s

    # gas information
    if "gas" in disk_ptypes:
        # only select gas particles within r_limit
        gas_tem_mask = np.log10(part["gas"]["temperature"]) < log_t_max

        if host_return_dict["use_host_prop"]:
            # only select gas within r_limit
            gas_pos_mask = part["gas"].prop("host.distance.principal.total") < r_limit

            gas_cyl_rad = part["gas"].prop("host.distance.principal.cylindrical")[:, 0][
                gas_pos_mask & gas_tem_mask
            ]
            gas_rot_vel = part["gas"].prop("host.velocity.principal.cylindrical")[:, 1][
                gas_pos_mask & gas_tem_mask
            ]

        else:
            halo_detail_dict = host_return_dict["halo_details"]

            # only select gas within r_limit
            gas_pos_mask = (
                ut.particle.get_distances_wrt_center(
                    part,
                    species=["gas"],
                    center_position=halo_detail_dict["position"],
                    rotation=halo_detail_dict["rotation"],
                    coordinate_system="cartesian",
                    total_distance=True,
                )
                < r_limit
            )

            gas_cyl_rad = ut.particle.get_distances_wrt_center(
                part,
                species=["gas"],
                center_position=halo_detail_dict["position"],
                rotation=halo_detail_dict["rotation"],
                coordinate_system="cylindrical",
                total_distance=False,
            )[:, 0][gas_pos_mask & gas_tem_mask]

            gas_rot_vel = ut.particle.get_velocities_wrt_center(
                part,
                species=["gas"],
                center_position=halo_detail_dict["position"],
                center_velocity=halo_detail_dict["velocity"],
                rotation=halo_detail_dict["rotation"],
                coordinate_system="cylindrical",
                total_velocity=False,
            )[:, 1][gas_pos_mask & gas_tem_mask]

        gas_mass = part["gas"]["mass"][gas_pos_mask & gas_tem_mask]

        v_rot_gas_arr = np.zeros(bin_num)
        sig_rot_gas_arr = np.zeros(bin_num)
        for i in range(bin_num):
            in_bin = (gas_cyl_rad >= bin_edges[i]) & (gas_cyl_rad < bin_edges[i + 1])
            if np.any(in_bin):
                weights = gas_mass[in_bin]
                v_rot_gas_i = np.average(gas_rot_vel[in_bin], weights=weights)
                v_rot_gas_i_2 = np.average(gas_rot_vel[in_bin] ** 2, weights=weights)

                sig_rot_gas_i = np.sqrt(v_rot_gas_i_2 - v_rot_gas_i**2)

                v_rot_gas_arr[i] = v_rot_gas_i
                sig_rot_gas_arr[i] = sig_rot_gas_i

            else:
                v_rot_gas_arr[i] = np.nan
                sig_rot_gas_arr[i] = np.nan

        v_rot_g = np.nanmax(v_rot_gas_arr)
        sig_rot_g = np.nanmedian(sig_rot_gas_arr)

        v_sig_dict["v_rot_g"] = v_rot_g
        v_sig_dict["sig_rot_g"] = sig_rot_g

    if ("star" in disk_ptypes) and ("gas" in disk_ptypes):
        cyl_rad_sg = np.concatenate((star_cyl_rad, gas_cyl_rad))
        rot_vel_sg = np.concatenate((star_rot_vel, gas_rot_vel))
        mass_sg = np.concatenate((star_mass, gas_mass))

        v_rot_sg_arr = np.zeros(bin_num)
        sig_rot_sg_arr = np.zeros(bin_num)
        for i in range(bin_num):
            in_bin = (cyl_rad_sg >= bin_edges[i]) & (cyl_rad_sg < bin_edges[i + 1])
            if np.any(in_bin):
                weights = mass_sg[in_bin]
                v_rot_sg_i = np.average(rot_vel_sg[in_bin], weights=weights)
                v_rot_sg_i_2 = np.average(rot_vel_sg[in_bin] ** 2, weights=weights)

                sig_rot_sg_i = np.sqrt(v_rot_sg_i_2 - v_rot_sg_i**2)

                v_rot_sg_arr[i] = v_rot_sg_i
                sig_rot_sg_arr[i] = sig_rot_sg_i

            else:
                v_rot_sg_arr[i] = np.nan
                sig_rot_sg_arr[i] = np.nan

        v_rot_sg = np.nanmax(v_rot_sg_arr)
        sig_rot_sg = np.nanmedian(sig_rot_sg_arr)

        v_sig_dict["v_rot_sg"] = v_rot_sg
        v_sig_dict["sig_rot_sg"] = sig_rot_sg

    return v_sig_dict


def make_images(
    part,
    halt,
    distance_max,
    distance_bin_width,
    main_halo_tid,
    sim,
    sim_dir,
    snapshot,
    species=["star", "gas", "dark"],
    weight_name="mass",
    image_limits=[None, None],
):
    dimen_label = {0: "x", 1: "y", 2: "z"}

    host_return_dict = get_halo_center(part, halt, main_halo_tid, sim, sim_dir, snapshot)

    # get distance limits for plot
    position_limits = [[-distance_max, distance_max] for _ in range(2)]
    position_limits = np.array(position_limits)

    # get number of bins (pixels) along each dimension
    position_bin_number = int(np.round(2 * np.max(distance_max) / distance_bin_width))

    num_rows = len(species)
    num_cols = 3

    fig_height = 6 * num_rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, fig_height))

    for i, spec in enumerate(species):
        if host_return_dict["use_host_prop"]:
            positions = part[spec].prop("host.distance.principal")

        else:
            halo_detail_dict = host_return_dict["halo_details"]

            positions = ut.particle.get_distances_wrt_center(
                part,
                species=[spec],
                center_position=halo_detail_dict["position"],
                rotation=halo_detail_dict["rotation"],
                coordinate_system="cartesian",
                total_distance=False,
            )

        weights = None
        if weight_name:
            weights = part[spec].prop(weight_name)

        # set color map
        if spec == "dark":
            color_map = plt.cm.afmhot
        elif spec == "gas":
            color_map = plt.cm.afmhot
        elif spec == "star":
            color_map = plt.cm.afmhot

        for plot_num in range(0, 3):
            if plot_num == 0:
                dims = [0, 1]
            if plot_num == 1:
                dims = [0, 2]
            if plot_num == 2:
                dims = [1, 2]

            # keep only particles within distance limits along each dimension
            masks = positions[:, dims[0]] <= distance_max
            for dim_i in dims:
                masks *= (positions[:, dim_i] >= -distance_max) * (positions[:, dim_i] <= distance_max)

            plot_pos = positions[masks]
            if weights is not None:
                plot_weights = weights[masks]

            hist_valuess, hist_xs, hist_ys = np.histogram2d(
                plot_pos[:, dims[0]],
                plot_pos[:, dims[1]],
                position_bin_number,
                position_limits,
                weights=plot_weights,
            )

            # convert to surface density
            hist_valuess /= np.diff(hist_xs)[0] * np.diff(hist_ys)[0]

            if num_rows == 1:
                ax = axs[plot_num]
            else:
                ax = axs[i, plot_num]

            ax.set_xlim(position_limits[0])
            ax.set_ylim(position_limits[1])

            ax.set_xlabel(f"{dimen_label[dims[0]]} $\\left[ {{\\rm kpc}} \\right]$")
            ax.set_ylabel(f"{dimen_label[dims[1]]} $\\left[ {{\\rm kpc}} \\right]$")

            # make 2-D histogram image
            Image = ax.imshow(
                hist_valuess.transpose(),
                norm=colors.LogNorm(),
                cmap=color_map,
                aspect="auto",
                interpolation="bilinear",
                extent=np.concatenate(position_limits),
                vmin=image_limits[0],
                vmax=image_limits[1],
            )

            fig.colorbar(Image)

            if plot_num == 1:
                ax.set_title(spec)

    save_path = sim_dir + sim + "/galaxy_images/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig_file = "snap_%d.png" % snapshot
    fig.savefig(save_path + fig_file)

    # plt.show(block=False)


def compile_kinematics(main_halo_tid: int, sim: str, sim_dir: str, snapshot: int):
    start_time = time.time()
    fire_dir = sim_dir + sim + "/" + sim + "_res7100/"

    global _global_halt
    halt = _global_halt  # safely read it

    use_stellar_radius = True
    stellar_radius_multiplier = 2
    r_max = None

    snap_halo_tid = gc_utils.get_halo_prog_at_snap(halt, main_halo_tid, snapshot)
    snap_halo_idx = np.where(halt["tid"] == snap_halo_tid)[0][0]

    if use_stellar_radius:
        r_limit = stellar_radius_multiplier * halt["star.radius.90"][snap_halo_idx]

    else:
        if r_max is None:
            raise RuntimeError("Need to provide r_max value if not using stellar radius")
        else:
            r_limit = r_max

    kin_dict = {}

    snap_id = gc_utils.snapshot_name(snapshot)
    kin_dict[snap_id] = {}
    kin_dict[snap_id]["r_limit"] = r_limit

    part = gc_utils.open_snapshot(snapshot, fire_dir, species=["all"])

    kappa_dict = get_kappa_co(halt, part, main_halo_tid, sim, sim_dir, snapshot, r_limit)
    kin_dict[snap_id].update(kappa_dict)

    sigma_dict = get_v_sigma(halt, part, main_halo_tid, sim, sim_dir, snapshot, r_limit)
    kin_dict[snap_id].update(sigma_dict)

    make_images(
        part,
        halt,
        r_limit * 1.5,
        0.1,
        main_halo_tid,
        sim,
        sim_dir,
        snapshot,
        species=["star", "gas", "dark"],
        weight_name="mass",
        image_limits=[None, None],
    )

    del part

    end_time = time.time()
    print(snapshot, "time:", end_time - start_time)

    return kin_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-l", "--location", required=True, type=str, help="either local or katana")
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
    location = args.location

    if location == "local":
        sim_dir = "../../simulations/"

    elif location == "katana":
        sim_dir = "/srv/scratch/astro/z5114326/simulations/"

    else:
        raise RuntimeError("Incorrect location provided. Must be local or katana.")

    # potential_snaps = sim_dir + sim + "/potentials.json"
    # with open(potential_snaps) as json_file:
    #     pot_data = json.load(json_file)

    cores = args.cores
    if cores is None:
        # 2 cores is max to run with 64 GB RAM
        # cores = mp.cpu_count()
        cores = 2

    file_path = sim_dir + sim + "/galaxy_kinematics.json"
    if not os.path.exists(file_path):
        # Step 2: Create it with an empty list
        with open(file_path, "w") as f:
            json.dump({}, f, indent=2)

    fire_dir = sim_dir + sim + "/" + sim + "_res7100/"

    print("Retrieving Halo Tree")
    gc_utils.block_print()
    halt = halo.io.IO.read_tree(simulation_directory=fire_dir, species="star")
    gc_utils.enable_print()
    print("Halo Tree Retrieved")

    sim_codes = sim_dir + "simulation_codes.json"
    with open(sim_codes) as sim_json:
        sim_data = json.load(sim_json)
    main_halo_tid = [sim_data[sim]["halo"]]

    snap_lst = args.snapshots
    if snap_lst is None:
        snap_lst = constant_star_snap(halt, main_halo_tid, sim_dir)

    print(snap_lst)

    with mp.Pool(processes=cores, maxtasksperchild=1, initializer=init_worker, initargs=(halt,)) as pool:
        results = pool.starmap(
            compile_kinematics, [(main_halo_tid, sim, sim_dir, snapshot) for snapshot in snap_lst]
        )

    del halt

    with open(file_path, "r+") as f:
        data = json.load(f)

        # combined_dict = {}
    for result in results:
        # combined_dict.update(result)
        data.update(result)

    # print(combined_dict)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
