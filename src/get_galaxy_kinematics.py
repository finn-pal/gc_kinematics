import json

import gc_utils
import halo_analysis as halo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utilities as ut


def constant_star_snap(halt, main_halo_tid: int, sim_dir: str = None, get_public_snap: bool = False):
    idx = np.where(halt["tid"] == main_halo_tid)[0][0]

    while halt["star.radius.90"][idx] > 0:
        star_snap = halt["snapshot"][idx]
        idx = halt["progenitor.main.index"][idx]

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

        star_snap = np.min(snap_lst[snap_lst >= star_snap])

    return star_snap


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


def get_densities(
    part,
    halt,
    main_halo_tid: int,
    sim: str,
    sim_dir: str,
    snapshot: int,
    r_limit: float,
    z_lim_fraction: float,
    disk_ptypes: list[str] = ["star", "gas"],
):
    host_return_dict = get_halo_center(part, halt, main_halo_tid, sim, sim_dir, snapshot)

    if host_return_dict["use_host_prop"]:
        host_pos = part.host["position"]
        host_rot = part.host["rotation"][0]

    else:
        halo_detail_dict = host_return_dict["halo_details"]
        host_pos = halo_detail_dict["position"]
        host_rot = halo_detail_dict["rotation"]

    SpeciesProfile = ut.particle.SpeciesProfileClass(
        limits=[0, r_limit], width=r_limit / 2, dimension_number=2
    )

    z_lim = r_limit * z_lim_fraction

    pro = SpeciesProfile.get_sum_profiles(
        part,
        disk_ptypes,
        "mass",
        center_position=host_pos,
        rotation=host_rot,
        other_axis_distance_limits=[-z_lim, z_lim],
    )

    mass_dict = {}

    if "star" in disk_ptypes:
        density_s_data = pro["star"]["density"]
        density_s_avg = np.average(density_s_data)
        density_s_std = np.std(density_s_data)

        mass_s_data = pro["star"]["sum"]
        mass_s_avg = np.average(mass_s_data)
        mass_s_std = np.std(mass_s_data)

        mass_dict["density_s_avg"] = density_s_avg
        mass_dict["density_s_std"] = density_s_std
        mass_dict["mass_s_avg"] = mass_s_avg
        mass_dict["mass_s_std"] = mass_s_std

    if "gas" in disk_ptypes:
        density_g_data = pro["gas"]["density"]
        density_g_avg = np.average(density_g_data)
        density_g_std = np.std(density_g_data)

        mass_g_data = pro["gas"]["sum"]
        mass_g_avg = np.average(mass_g_data)
        mass_g_std = np.std(mass_g_data)

        mass_dict["density_g_avg"] = density_g_avg
        mass_dict["density_g_std"] = density_g_std
        mass_dict["mass_g_avg"] = mass_g_avg
        mass_dict["mass_g_std"] = mass_g_std

    if ("star" in disk_ptypes) and ("gas" in disk_ptypes):
        density_sg_data = pro["baryon"]["density"]
        density_sg_avg = np.average(density_sg_data)
        density_sg_std = np.std(density_sg_data)

        mass_sg_data = pro["baryon"]["sum"]
        mass_sg_avg = np.average(mass_sg_data)
        mass_sg_std = np.std(mass_sg_data)

        mass_dict["density_sg_avg"] = density_sg_avg
        mass_dict["density_sg_std"] = density_sg_std
        mass_dict["mass_sg_avg"] = mass_sg_avg
        mass_dict["mass_sg_std"] = mass_sg_std

    return mass_dict
