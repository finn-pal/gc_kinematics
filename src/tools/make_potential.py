import json
import os
import time
import warnings

import agama
import gc_utils
import gizmo_analysis as gizmo
import halo_analysis as halo
import matplotlib.pyplot as plt
import numpy as np
import utilities as ut

# from gc_utils import block_print, enable_print, get_main_vir_rad_snap  # type: ignore

# the below function "make_potential" is derived from that of "fitPotential" presented in the AGAMA package
# under py -> "example_gizmo_snapshot"

# tunable parameters for the potentials are:
# GridSizeR, rmin, rmax - specify the (logarithmic) grid in spherical radius
# (for Multipole) or cylindrical radius (for CylSpline); for the latter, there is a second grid in z,
# defined by GridSizeZ, zmin, zmax (by default the same as the radial grid).
# The minimum radius should be comparable to the smallest resolvable feature in the snapshot
# (e.g., the extent of the central cusp or core, or the disk scale height), but not smaller.
# The maximum radius defines the region where the potential and density approximated in full detail;
# note that for Multipole, the density is extrapolated as a power law outside this radius,
# and the potential takes into account the contribution of input particles at all radii,
# while for CylSpline the particles outside the grid are ignored, and the density is zero there.
# lmax - the order of multipole expansion;
# mmax - the order of azimuthal Fourier expansion for CylSpline (if it is axisymmetric, mmax is ignored)
# Note that the default parameters are quite sensible, but not necessarily optimal for your case.


def make_potential(
    halt,
    sim: str,
    snapshot: int,
    sim_dir: str,
    disk_limt: int = None,
    # rmax_sel: int = 600,  # set as 2x virial radius
    # rmax_ctr: int = 10,  # set as 0.05x virial radius (minimum 1)
    # rmax_exp: int = 500,  # set as 1.5x virial radius
    rmax_sel_frac: float = 2,  # set as 2x virial radius
    rmax_ctr_frac: float = 0.05,  # set as 0.05x virial radius (minimum 1)
    rmax_exp_frac: float = 1.5,  # set as 1.5x virial radius
    symmetry: str = "a",  # needs to be "a" for orbit propagation
    subsample_factor: int = 1,
    save_coords: bool = True,
    save_plot: bool = True,
    print_plot: bool = False,
    compare_plot: bool = True,
    verbose: bool = False,
):
    print(snapshot)
    start_time = time.time()

    sim_codes = sim_dir + "simulation_codes.json"
    with open(sim_codes) as sim_json:
        sim_data = json.load(sim_json)
    main_halo_tid = [sim_data[sim]["halo"]]

    # get rmax_sel, rmax_ctr, rmax_exp

    # fire simulation location
    fire_dir = sim_dir + sim + "/" + sim + "_res7100/"

    # # block printing
    # gc_utils.block_print()
    # halt = halo.io.IO.read_tree(simulation_directory=fire_dir)
    # # enable printing
    # gc_utils.enable_print()

    snap_halo_tid = gc_utils.get_halo_prog_at_snap(halt, main_halo_tid, snapshot)
    halo_idx = np.where(halt["tid"] == snap_halo_tid)[0][0]
    halo_radius = halt["radius"][halo_idx]

    rmax_sel = halo_radius * rmax_sel_frac
    rmax_ctr = halo_radius * rmax_ctr_frac
    rmax_exp = halo_radius * rmax_exp_frac

    if rmax_ctr < 1:
        rmax_ctr = 1

    # define the physical units used in the code: the choice below corresponds to
    # length scale = 1 kpc, velocity = 1 km/s, mass = 1 Msun
    agama.setUnits(mass=1, length=1, velocity=1)

    # particles of interest
    ptypes = ["gas", "star", "dark"]

    if verbose:
        print("reading snapshot")

    # block printing
    gc_utils.block_print()

    # read in the snapshot
    part = gizmo.io.Read.read_snapshots(
        species=ptypes,
        snapshot_value_kind="index",
        snapshot_values=snapshot,
        simulation_directory=fire_dir,
        particle_subsample_factor=subsample_factor,
        assign_hosts_rotation=True,
    )

    # enable printing
    gc_utils.enable_print()

    ###################################################################################################
    # update 08/04/2024
    # this is to fix the centring of the potenital on the progenitor of the host at z = 0 rather than the
    # host at the relevant snapshot.

    # block printing
    gc_utils.block_print()
    halt = halo.io.IO.read_tree(simulation_directory=fire_dir)
    # enable printing
    gc_utils.enable_print()

    halt_center_snap_lst = sim_data[sim]["dm_center"]
    use_dm_center = snapshot in halt_center_snap_lst

    # check to see is the progentior of the host at z = 0 is the host at this snapshot
    not_host_snap_lst = gc_utils.get_different_snap_lst(main_halo_tid, halt, sim, sim_dir)
    is_main_host = snapshot not in not_host_snap_lst

    if (not is_main_host) or (use_dm_center):
        halo_tid = gc_utils.get_main_prog_at_snap(halt, main_halo_tid, snapshot)

        if use_dm_center:
            halo_details = gc_utils.get_dm_halo_details(part, halt, halo_tid, snapshot, True)
        else:
            halo_details = gc_utils.get_halo_details(part, halt, halo_tid, snapshot)

        # start with default centering and rotation to define aperture
        dist = ut.particle.get_distances_wrt_center(
            part,
            species=ptypes,
            center_position=halo_details["position"],
            rotation=halo_details["rotation"],
            total_distance=True,
        )

        # i dont think i need distance vectors here and also i dont think rotation matters in this first part
        # dist_vectors = ut.particle.get_distances_wrt_center(
        #     part,
        #     species=ptypes,
        #     center_position=halo_details["position"],
        #     rotation=halo_details["rotation"],
        # )

    else:
        # start with default centering and rotation to define aperture
        dist = ut.particle.get_distances_wrt_center(
            part, species=ptypes, center_position=part.host["position"], rotation=True, total_distance=True
        )

        # dist_vectors = ut.particle.get_distances_wrt_center(
        #     part, species=ptypes, center_position=part.host["position"], rotation=True
        # )

    # compute new centering and rotation using a fixed aperture in stars
    sp = "star"
    ctr_indices = np.where(dist[sp] < rmax_ctr)[0]

    if len(ctr_indices) > 0:
        m = part[sp]["mass"][ctr_indices]
        pos = part[sp]["position"][ctr_indices]
        vel = part[sp]["velocity"][ctr_indices]
        new_ctr = np.multiply(m, pos.T).sum(axis=1) / np.sum(m)
        new_vctr = np.multiply(m, vel.T).sum(axis=1) / np.sum(m)

    else:
        sp = "dark"
        ctr_indices = np.where(dist[sp] < rmax_ctr)[0]
        m = part[sp]["mass"][ctr_indices]
        pos = part[sp]["position"][ctr_indices]
        vel = part[sp]["velocity"][ctr_indices]
        new_ctr = np.multiply(m, pos.T).sum(axis=1) / np.sum(m)
        new_vctr = np.multiply(m, vel.T).sum(axis=1) / np.sum(m)

    # block printing
    gc_utils.block_print()

    new_rot = ut.particle.get_principal_axes(
        part, species_name=sp, part_indicess=ctr_indices, center_positions=new_ctr, center_velocities=new_vctr
    )

    # enable printing
    gc_utils.enable_print()

    # optionally compute acceleration of center of mass frame if it was recorded
    save_accel = "acceleration" in part[sp].keys()

    if save_accel:
        if verbose:
            print("saving acceleration of COM frame")
        accel = part[sp]["acceleration"][ctr_indices]
        new_actr = np.multiply(m, accel.T).sum(axis=1) / np.sum(m)

    # recompute distances in the new frame

    dist = ut.particle.get_distances_wrt_center(
        part, species=ptypes, center_position=new_ctr, rotation=new_rot["rotation"], total_distance=True
    )

    dist_vectors = ut.particle.get_distances_wrt_center(
        part, species=ptypes, center_position=new_ctr, rotation=new_rot["rotation"]
    )

    # pick out gas and stars within the region that we want to supply to the model

    m_gas_tot = part["gas"]["mass"].sum() * subsample_factor

    # pos_pa_gas = dist_vectors["gas"][dist["gas"] < rmax_sel]
    m_gas = part["gas"]["mass"][dist["gas"] < rmax_sel] * subsample_factor
    if verbose:
        print("{0:.3g} of {1:.3g} solar masses in gas selected".format(m_gas.sum(), m_gas_tot))

    m_star_tot = part["star"]["mass"].sum() * subsample_factor

    pos_pa_star = dist_vectors["star"][dist["star"] < rmax_sel]
    m_star = part["star"]["mass"][dist["star"] < rmax_sel] * subsample_factor
    if verbose:
        print("{0:.3g} of {1:.3g} solar masses in stars selected".format(m_star.sum(), m_star_tot))

    # separate cold gas in disk (modeled with cylspline) from hot gas in halo (modeled with multipole)

    tsel = np.log10(part["gas"]["temperature"]) < 4.5

    rsel = dist["gas"] < rmax_sel

    pos_pa_gas_cold = dist_vectors["gas"][tsel & rsel]
    m_gas_cold = part["gas"]["mass"][tsel & rsel] * subsample_factor
    if verbose:
        print(
            "{0:.3g} of {1:.3g} solar masses are cold gas to be modeled with cylspline".format(
                m_gas_cold.sum(), m_gas.sum()
            )
        )

    pos_pa_gas_hot = dist_vectors["gas"][(~tsel) & rsel]
    m_gas_hot = part["gas"]["mass"][(~tsel) & rsel] * subsample_factor
    if verbose:
        print(
            "{0:.3g} of {1:.3g} solar masses are hot gas to be modeled with multipole".format(
                m_gas_hot.sum(), m_gas.sum()
            )
        )

    # combine components that will be fed to the cylspline part
    pos_pa_bar = np.vstack((pos_pa_star, pos_pa_gas_cold))
    m_bar = np.hstack((m_star, m_gas_cold))

    # pick out the dark matter
    m_dark_tot = part["dark"]["mass"].sum() * subsample_factor

    rsel = dist["dark"] < rmax_sel
    pos_pa_dark = dist_vectors["dark"][rsel]
    m_dark = part["dark"]["mass"][rsel] * subsample_factor
    if verbose:
        print("{0:.3g} of {1:.3g} solar masses in dark matter selected".format(m_dark.sum(), m_dark_tot))

    # stack with hot gas for multipole density
    pos_pa_dark = np.vstack((pos_pa_dark, pos_pa_gas_hot))
    m_dark = np.hstack((m_dark, m_gas_hot))

    # save_dir = data_dir + "potentials/" + sim + "/snap_%d/" % snapshot  # save location
    save_dir = sim_dir + sim + "/potentials" + "/snap_%d/" % snapshot  # save location
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_coords:
        # save the Hubble parameter to transform to comoving units
        hubble = part.info["hubble"]
        scalefactor = part.info["scalefactor"]

        cname = "{0}_coords.txt".format(save_dir)

        if verbose:
            print("Saving coordinate transformation to {0}".format(cname))

        with open(cname, "w") as f:
            f.write("# Hubble parameter and scale factor (to convert physical <-> comoving) \n")
            f.write("{0:.18g} {1:.18g}\n".format(hubble, scalefactor))
            f.write("# center position (kpc comoving)\n")
            np.savetxt(f, new_ctr)
            f.write("# center velocity (km/s physical)\n")
            np.savetxt(f, new_vctr)
            if save_accel:
                f.write("# center acceleration (km/s^2 physical)\n")
                np.savetxt(f, new_actr)
            f.write("# rotation to principal-axis frame\n")
            np.savetxt(f, new_rot["rotation"])

    if verbose:
        print("Computing multipole expansion coefficients for dark matter/hot gas component")

    # get potential for dark matter and hot gas via multipole
    p_dark = agama.Potential(
        type="multipole",
        particles=(pos_pa_dark, m_dark),
        lmax=4,  # changed from 2
        # mmax=0, # commented out
        gridSizeR=20,  # was 20
        symmetry=symmetry,
        rmin=0.1,
        rmax=rmax_exp,
    )

    # save halo potential
    pot_halo = "halo_snap_%d.ini" % snapshot
    p_dark.export(save_dir + pot_halo)

    if verbose:
        print("Computing cylindrical spline coefficients for stellar/cold gas component")

    # get potential for stars and cold gas via cylspline

    if disk_limt is None:
        p_disk = agama.Potential(
            type="cylspline",
            particles=(pos_pa_bar, m_bar),
            mmax=4,  # changed from 0
            symmetry=symmetry,
            gridsizer=20,  # was 20
            gridsizez=20,  # was 20
            rmin=0.1,
            rmax=rmax_exp,
        )

    else:
        if snapshot >= disk_limt:
            # if snapshot is after disk formation plot as disk
            p_disk = agama.Potential(
                type="cylspline",
                particles=(pos_pa_bar, m_bar),
                mmax=4,  # changed from 0
                symmetry=symmetry,
                gridsizer=20,  # was 20
                gridsizez=20,  # was 20
                rmin=0.1,
                rmax=rmax_exp,
            )

        else:
            # else plot as multipole
            p_disk = agama.Potential(
                type="multipole",
                particles=(pos_pa_bar, m_bar),
                lmax=4,  # changed from 2
                # mmax=0, # commented out
                gridSizeR=20,  # was 20
                symmetry=symmetry,
                rmin=0.1,
                rmax=rmax_exp,
            )

    # save disk potential
    pot_disk = "disk_snap_%d.ini" % snapshot
    p_disk.export(save_dir + pot_disk)

    # combine potentials
    pot_nbody = agama.Potential(p_dark, p_disk)

    # save combined potential
    pot_comb = "combined_snap_%d.ini" % snapshot
    pot_nbody.export(save_dir + pot_comb)

    if save_plot or print_plot:
        if verbose:
            print("Making plots")
        plot_potential(pot_nbody, halt, main_halo_tid, sim, snapshot, save_dir, save_plot, print_plot)
        relative_errors(pot_nbody, part, halt, sim, snapshot, sim_dir, save_dir, save_plot, print_plot)

        if not print_plot:
            plt.close()

        if compare_plot:
            compare_potentials(
                pot_nbody,
                part,
                halt,
                sim,
                snapshot,
                sim_dir=sim_dir,
                save_dir=save_dir,
                save_plot=save_plot,
                print_plot=print_plot,
            )

        if not print_plot:
            plt.close()

    end_time = time.time()
    print(snapshot, "time:", end_time - start_time)

    del part
    del halt


def plot_potential(
    pot_nbody,
    halt,
    main_halo_tid: int,
    sim: str,
    snapshot: int = None,
    save_dir: str = None,
    save_plot: bool = False,
    print_plot: bool = True,
):
    max_rad = gc_utils.get_main_vir_rad_snap(halt, main_halo_tid, snapshot)
    max_grid = 2 * int(max_rad)

    gridR = agama.nonuniformGrid(500, 0.01, max_grid)
    gridz = agama.symmetricGrid(500, 0.01, max_grid)
    gridR00 = np.column_stack((gridR, gridR * 0, gridR * 0))
    grid00z = np.column_stack((gridz * 0, gridz * 0, gridz))

    # xticks_density = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].loglog(gridR, pot_nbody.density(gridR00), label="Agama", color="blue")
    # axs[0, 0].set(xlim=[1e-2, max_grid], xticks=xticks_density)
    axs[0, 0].set(xlim=[1e-2, max_grid])
    axs[0, 0].set_xlabel("R (kpc)")
    axs[0, 0].set_ylabel(r"$\rho(R)$ at $z=0$ $[M_\odot/\mathsf{kpc}^3]$")

    axs[0, 1].loglog(gridz, pot_nbody.density(grid00z), label="Agama", color="blue")
    # axs[0, 1].set(xlim=[1e-2, max_grid], xticks=xticks_density[1:])
    axs[0, 1].set(xlim=[1e-2, max_grid])
    axs[0, 1].set_xlabel("z (kpc)")
    axs[0, 1].set_ylabel(r"$\rho(z)$ $[M_\odot/\mathsf{kpc}^3]$")

    axs[1, 0].plot(gridR, pot_nbody.potential(gridR00) * 10**-5, label="Agama", color="blue")
    axs[1, 0].set(xlim=[0, max_grid])
    axs[1, 0].set_xlabel("R (kpc)")
    axs[1, 0].set_ylabel("E$_{p}(R)$ at $z=0$ (10$^{5}$ km$^{2}$ s$^{-2}$)")

    axs[1, 1].plot(gridz, pot_nbody.potential(grid00z) * 10**-5, label="Agama", color="blue")
    axs[1, 1].set(xlim=[-max_grid, max_grid])
    axs[1, 1].set_xlabel("z (kpc)")
    axs[1, 1].set_ylabel("E$_{p}(R)$ (10$^{5}$ km$^{2}$ s$^{-2}$)")

    plt.suptitle(sim + " Snapshot %d" % snapshot)

    if print_plot:
        plt.show()

    if save_plot:
        if save_dir is None:
            print("No save directory provided, figure not saved")
        else:
            fig_file = "snap_%d.png" % snapshot
            fig.savefig(save_dir + fig_file)


def compare_potentials(
    pot_nbody,
    part,
    halt,
    sim: str,
    snapshot: int,
    pos_limit_width: float = 1,
    z_width: float = 1,
    r_width: float = 2,
    bin_width: float = 1,
    sim_dir: str = None,  # this shouldn't be None
    save_dir: str = None,  # this shouldn't be None
    save_plot: bool = False,
    print_plot: bool = True,
    save_offset: bool = True,
):
    # to avoid the empty array warnings
    np.seterr(divide="ignore", invalid="ignore")

    # Suppress Python RuntimeWarnings (like "Mean of empty slice")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # save_dir = data_dir + "potentials/" + sim + "/snap_%d/" % snapshot  # save location
    # save_dir = sim_dir + sim + "potentials" + "/snap_%d/" % snapshot  # save location

    sim_codes = sim_dir + "simulation_codes.json"
    with open(sim_codes) as sim_json:
        sim_data = json.load(sim_json)
    main_halo_tid = [sim_data[sim]["halo"]]

    halt_center_snap_lst = sim_data[sim]["dm_center"]
    use_dm_center = snapshot in halt_center_snap_lst

    # host_tid = [data[sim]["halo"]]
    # host_idx = np.where(halt["tid"] == host_tid)[0][0]

    #### I think the way I am defining host name is wrong host_index == 0 always
    # host_name = ut.catalog.get_host_name(0)

    max_rad = gc_utils.get_main_vir_rad_snap(halt, main_halo_tid, snapshot)
    max_rad = int(max_rad)

    gridR = np.linspace(0, max_rad, int(10 * max_rad) + 1)
    gridz = agama.symmetricGrid(10 * max_rad + 1, 0.01, max_rad)
    gridR00 = np.column_stack((gridR, gridR * 0, gridR * 0))
    grid00z = np.column_stack((gridz * 0, gridz * 0, gridz))

    ptype_lst = ["star", "gas", "dark"]

    r_lim_min = max_rad - pos_limit_width / 2
    r_lim_max = max_rad + pos_limit_width / 2
    z_min = -z_width / 2
    z_max = z_width / 2

    # check to see is the progentior of the host at z = 0 is the host at this snapshot
    not_host_snap_lst = gc_utils.get_different_snap_lst(main_halo_tid, halt, sim, sim_dir)
    is_main_host = snapshot not in not_host_snap_lst

    if (not is_main_host) or (use_dm_center):
        halo_tid = gc_utils.get_main_prog_at_snap(halt, main_halo_tid, snapshot)

        if use_dm_center:
            halo_details = gc_utils.get_dm_halo_details(part, halt, halo_tid, snapshot, True)
        else:
            halo_details = gc_utils.get_halo_details(part, halt, halo_tid, snapshot)

    potential_lst = np.array([])

    for ptype in ptype_lst:
        if (not is_main_host) or (use_dm_center):
            cyl = ut.particle.get_distances_wrt_center(
                part,
                species=[ptype],
                center_position=halo_details["position"],
                rotation=halo_details["rotation"],
                coordinate_system="cylindrical",
            )

        else:
            cyl = part[ptype].prop("host.distance.principal.cylindrical")

        mask = (
            (r_lim_min <= cyl[:, 0]) & (cyl[:, 0] <= r_lim_max) & (z_min <= cyl[:, 2]) & (cyl[:, 2] <= z_max)
        )

        potentials = np.array(part[ptype]["potential"][mask])
        potential_lst = np.concatenate((potential_lst, potentials))

    potential_offset = np.average(potential_lst) - pot_nbody.potential(gridR00[-1])

    ##########################################################################################

    # for ptype in ptype_lst:
    #     if (not is_main_host) or (use_dm_center):
    #         dist = ut.particle.get_distances_wrt_center(
    #             part,
    #             species=[ptype],
    #             center_position=halo_details["position"],
    #             rotation=halo_details["rotation"],
    #             coordinate_system="cartesian",
    #             total_distance=True,
    #         )

    #     else:
    #         dist = part[ptype].prop("host.distance.principal.total")

    #     mask = dist <= pos_limit_width / 2

    #     potentials = np.array(part[ptype]["potential"][mask])
    #     potential_lst = np.concatenate((potential_lst, potentials))

    # potential_offset = np.average(potential_lst) - pot_nbody.potential(gridR00[0])

    ##########################################################################################

    r_position_list = np.array([])
    r_potential_list = np.array([])

    z_position_list = np.array([])
    z_potential_list = np.array([])

    for ptype in ptype_lst:
        if (not is_main_host) or (use_dm_center):
            cyl = ut.particle.get_distances_wrt_center(
                part,
                species=[ptype],
                center_position=halo_details["position"],
                rotation=halo_details["rotation"],
                coordinate_system="cylindrical",
            )

        else:
            cyl = part[ptype].prop("host.distance.principal.cylindrical")

        r_mask = (cyl[:, 0] <= max_rad) & (z_min <= cyl[:, 2]) & (cyl[:, 2] <= z_max)
        r_cyl_mask = cyl[:, 0][r_mask]
        r_position_list = np.concatenate((r_position_list, r_cyl_mask))

        r_potentials = np.array(part[ptype]["potential"][r_mask]) - potential_offset
        r_potential_list = np.concatenate((r_potential_list, r_potentials))

        z_mask = (np.abs(cyl[:, 2]) <= max_rad) & (cyl[:, 0] <= r_width)
        z_cyl_mask = cyl[:, 2][z_mask]
        z_position_list = np.concatenate((z_position_list, z_cyl_mask))

        z_potentials = np.array(part[ptype]["potential"][z_mask]) - potential_offset
        z_potential_list = np.concatenate((z_potential_list, z_potentials))

    r_bins = np.linspace(0, max_rad, int(max_rad / bin_width) + 1)
    z_bins = np.linspace(-max_rad, max_rad, 2 * int(max_rad / bin_width) + 1)

    r_agama_lst = []
    r_fire_lst = []

    z_agama_lst = []
    z_fire_lst = []

    for r_bin in r_bins[:-1]:
        r_bin_min = r_bin
        r_bin_max = r_bin + bin_width

        r_agama_mask = (r_bin_min < gridR) & (gridR <= r_bin_max)
        r_agama_potentials = pot_nbody.potential(gridR00)
        r_agama_potentials_mask = r_agama_potentials[r_agama_mask]
        r_agama_avg = np.average(r_agama_potentials_mask)
        r_agama_lst.append(r_agama_avg)

        r_fire_mask = (r_bin_min < r_position_list) & (r_position_list <= r_bin_max)
        r_fire_potenitals_mask = r_potential_list[r_fire_mask]
        r_fire_avg = np.average(r_fire_potenitals_mask)
        r_fire_lst.append(r_fire_avg)

    for z_bin in z_bins[:-1]:
        z_bin_min = z_bin
        z_bin_max = z_bin + bin_width

        z_agama_mask = (z_bin_min < gridz) & (gridz <= z_bin_max)
        z_agama_potentials = pot_nbody.potential(grid00z)
        z_agama_potentials_mask = z_agama_potentials[z_agama_mask]
        z_agama_avg = np.average(z_agama_potentials_mask)
        z_agama_lst.append(z_agama_avg)

        z_fire_mask = (z_bin_min < z_position_list) & (z_position_list <= z_bin_max)
        z_fire_potenitals_mask = z_potential_list[z_fire_mask]
        z_fire_avg = np.average(z_fire_potenitals_mask)
        z_fire_lst.append(z_fire_avg)

    r_error_lst = np.abs(np.array(r_fire_lst) - np.array(r_agama_lst)) / np.abs(np.array(r_fire_lst))
    z_error_lst = np.abs(np.array(z_fire_lst) - np.array(z_agama_lst)) / np.abs(np.array(z_fire_lst))
    error_max = np.nanmax(np.concatenate((r_error_lst, z_error_lst)))
    error_max_rnd = np.round(error_max, 2)
    error_bins = np.linspace(0, error_max_rnd, int(error_max_rnd * 100 + 1))

    fig = plt.figure(figsize=(15, 10))

    gs = fig.add_gridspec(2, 2, width_ratios=(1, 1), height_ratios=(4, 3))

    gs00 = gs[0, 0].subgridspec(2, 1, hspace=0, height_ratios=(3, 1))
    gs01 = gs[1, 0].subgridspec(1, 1)

    gs10 = gs[0, 1].subgridspec(2, 1, hspace=0, height_ratios=(3, 1))
    gs11 = gs[1, 1].subgridspec(1, 1)

    ax00, ax01 = gs00.subplots(sharex=True)
    ax02 = gs01.subplots()

    ax10, ax11 = gs10.subplots(sharex=True)
    ax12 = gs11.subplots()

    # radial

    ax00.plot(r_bins[:-1], np.array(r_fire_lst) * 10**-5, c="r", label="Average Fire Potential")
    ax00.plot(r_bins[:-1], np.array(r_agama_lst) * 10**-5, c="b", label="Average Agama Potential")
    ax00.set_ylabel("E$_{p}(R)$ at $z=0$ (10$^{5}$ km$^{2}$ s$^{-2}$)")
    ax00.tick_params(axis="x", direction="in")
    ax00.legend()

    ax01.plot(r_bins[:-1], r_error_lst)
    ax01.set_xlabel("R (kpc)")
    ax01.set_ylabel("Relative Error")
    r_lim_max = np.ceil(np.nanmax(z_error_lst) * 100) / 100.0
    # ax01.set_ylim(0, r_lim_max)

    ax02.hist(r_error_lst, bins=error_bins, histtype="step")
    # ax02.set_xticks(np.linspace(0, error_max_rnd, 11))
    ax02.set_xlabel("Relative Error")
    ax02.set_ylabel("Bin Count")

    # z direction

    ax10.plot(z_bins[:-1], np.array(z_fire_lst) * 10**-5, c="r", label="Average Fire Potential")
    ax10.plot(z_bins[:-1], np.array(z_agama_lst) * 10**-5, c="b", label="Average Agama Potential")
    ax10.set_ylabel("E$_{p}(z)$ (10$^{5}$ km$^{2}$ s$^{-2}$)")
    ax10.tick_params(axis="x", direction="in")

    ax11.plot(z_bins[:-1], z_error_lst)
    ax11.set_xlabel("z (kpc)")
    ax11.set_ylabel("Relative Error")
    # ax11.set_ylim(0, np.round(np.nanmax(z_error_lst), 2))
    # z_lim_max = np.ceil(np.nanmax(z_error_lst) * 100) / 100.0
    # ax11.set_ylim(0, z_lim_max)

    ax12.hist(z_error_lst, bins=error_bins, histtype="step")
    # ax12.set_xticks(np.linspace(0, error_max_rnd, 11))
    ax12.set_xlabel("Relative Error")
    ax12.set_ylabel("Bin Count")

    plt.suptitle(sim + " Snapshot %d" % snapshot)

    if print_plot:
        plt.show()

    if save_plot:
        if save_dir is None:
            print("No save directory provided, figure not saved")
        else:
            fig_file = "snap_%d_pot_comp.png" % snapshot
            fig.savefig(save_dir + fig_file)

    if save_offset:
        save_offset

        oname = "{0}_potential_offset.txt".format(save_dir)

        with open(oname, "w") as f:
            f.write("# sim " + sim + "\n")
            f.write("# snapshot %d \n" % snapshot)
            f.write("# potential offset \n")
            np.savetxt(f, [potential_offset])


def relative_errors(
    pot_nbody,
    part,
    halt,
    sim: str,
    snapshot: int,
    sim_dir: str = None,  # this shouldn't be None
    save_dir: str = None,  # this shouldn't be None
    save_plot: bool = False,
    print_plot: bool = True,
    save_offset: bool = True,
):
    ptype_lst = ["star", "gas", "dark"]

    sim_codes = sim_dir + "simulation_codes.json"
    with open(sim_codes) as sim_json:
        sim_data = json.load(sim_json)
    main_halo_tid = [sim_data[sim]["halo"]]

    halt_center_snap_lst = sim_data[sim]["dm_center"]
    use_dm_center = snapshot in halt_center_snap_lst

    # check to see is the progentior of the host at z = 0 is the host at this snapshot
    not_host_snap_lst = gc_utils.get_different_snap_lst(main_halo_tid, halt, sim, sim_dir)
    is_main_host = snapshot not in not_host_snap_lst

    if (not is_main_host) or (use_dm_center):
        halo_tid = gc_utils.get_main_prog_at_snap(halt, main_halo_tid, snapshot)

        if use_dm_center:
            halo_details = gc_utils.get_dm_halo_details(part, halt, halo_tid, snapshot, True)
        else:
            halo_details = gc_utils.get_halo_details(part, halt, halo_tid, snapshot)

    max_rad = gc_utils.get_main_vir_rad_snap(halt, main_halo_tid, snapshot)
    r_max = int(max_rad)

    pos_vect_lst = np.empty((0, 3))
    pos_dist_lst = np.array([])
    fire_pot_lst = np.array([])

    for ptype in ptype_lst:
        if (not is_main_host) or (use_dm_center):
            pos_vect = ut.particle.get_distances_wrt_center(
                part,
                species=[ptype],
                center_position=halo_details["position"],
                rotation=halo_details["rotation"],
                coordinate_system="cartesian",
            )

            pos_dist = ut.particle.get_distances_wrt_center(
                part,
                species=[ptype],
                center_position=halo_details["position"],
                rotation=halo_details["rotation"],
                coordinate_system="cartesian",
                total_distance=True,
            )

        else:
            pos_vect = part[ptype].prop("host.distance.principal")
            pos_dist = part[ptype].prop("host.distance.principal.total")

        mask = pos_dist < r_max

        pos_vect = pos_vect[mask]
        pos_dist = pos_dist[mask]
        potential = part[ptype]["potential"][mask]

        pos_vect_lst = np.vstack([pos_vect_lst, pos_vect])
        pos_dist_lst = np.concatenate([pos_dist_lst, pos_dist])
        fire_pot_lst = np.concatenate([fire_pot_lst, potential])

    agama_potentials = pot_nbody.potential(pos_vect_lst)
    offset = np.mean(fire_pot_lst - agama_potentials)

    # bin_size = 1
    bin_num = 100
    bin_edges = np.linspace(0, r_max, bin_num + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_err_avg_arr = np.zeros(bin_num)
    bin_err_std_arr = np.zeros(bin_num)
    for i in range(bin_num):
        in_bin = (pos_dist_lst >= bin_edges[i]) & (pos_dist_lst < bin_edges[i + 1])
        fire_bin_pot = fire_pot_lst[in_bin] - offset
        agama_bin_pot = agama_potentials[in_bin]

        bin_err = np.abs(fire_bin_pot - agama_bin_pot) / np.abs(fire_bin_pot)
        bin_err_avg = np.mean(bin_err)
        bin_err_std = np.std(bin_err)

        bin_err_avg_arr[i] = bin_err_avg
        bin_err_std_arr[i] = bin_err_std

    lower = bin_err_avg_arr - bin_err_std_arr
    upper = bin_err_avg_arr + bin_err_std_arr

    fig = plt.figure(figsize=(15, 10))

    plt.plot(bin_centers, bin_err_avg_arr, c="b", label="Average Error")
    plt.fill_between(bin_centers, lower, upper, color="b", alpha=0.3, label="Error Spread")

    plt.plot([0, r_max], [0.05, 0.05], c="grey", ls="--", label="5% Relative Error")

    plt.xlabel("Radius [kpc]")
    plt.ylabel("Relative Error of Agama Potential")

    plt.xlim([0, r_max])

    y_max = np.min([1, 1.5 * np.max(upper)])
    plt.ylim([0, y_max])

    snap_id = gc_utils.snapshot_name(snapshot)
    plt.title(snap_id + "\n" + "summed over all particles within r_max")
    plt.legend(loc="upper left")

    if print_plot:
        plt.show()

    if save_plot:
        if save_dir is None:
            print("No save directory provided, figure not saved")
        else:
            fig_file = "snap_%d_relative_error.png" % snapshot
            fig.savefig(save_dir + fig_file)

    if save_offset:
        oname = "{0}_potential_offset_better.txt".format(save_dir)

        with open(oname, "w") as f:
            f.write("# sim " + sim + "\n")
            f.write("# snapshot %d \n" % snapshot)
            f.write("# potential offset \n")
            np.savetxt(f, [offset])
