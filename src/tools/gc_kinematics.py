import agama
import h5py
import numpy as np
from gc_utils import iteration_name, snapshot_name  # type: ignore


def get_kinematics(
    sim: str,
    it_lst: list[int],
    snapshot: int,
    sim_dir: str,
    data_dict: dict = {},
):
    snap_id = snapshot_name(snapshot)
    print(snap_id)

    proc_file = sim_dir + sim + "/" + sim + "_processed.hdf5"
    potential_file = sim_dir + sim + "/potentials/snap_%d/combined_snap_%d.ini" % (snapshot, snapshot)

    proc_data = h5py.File(proc_file, "r")  # open processed data file

    agama.setUnits(mass=1, length=1, velocity=1)
    pot_nbody = agama.Potential(potential_file)
    af = agama.ActionFinder(pot_nbody, interp=False)

    it_dict = {}
    for it in it_lst:
        it_id = iteration_name(it)

        snap_data = proc_data[it_id]["snapshots"][snap_id]
        gc_id_snap = snap_data["gc_id"]

        # x = np.array(snap_data["x"])
        # y = np.array(snap_data["y"])
        # z = np.array(snap_data["z"])

        pos = snap_data["pos.xyz"][()]
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        # vx = np.array(snap_data["vx"])
        # vy = np.array(snap_data["vy"])
        # vz = np.array(snap_data["vz"])

        vel = snap_data["vel.xyz"][()]
        vx = vel[:, 0]
        vy = vel[:, 1]
        vz = vel[:, 2]

        l_xyz = snap_data["l.xyz"][()]
        l_mag = np.linalg.norm(l_xyz, axis=1)
        l_z = l_xyz[:, 2]
        inclination = np.degrees(np.arccos(l_z / l_mag))

        ek = np.array(snap_data["ek"])

        if len(gc_id_snap) is None:
            continue

        init_cond_lst = np.vstack((x, y, z, vx, vy, vz)).T

        ioms = af(init_cond_lst)
        jr = ioms[:, 0]
        jz = ioms[:, 1]
        jphi = ioms[:, 2]

        j_cyl = np.column_stack((jr, jphi, jz))

        radii_pos = pot_nbody.Rperiapo(init_cond_lst)
        r_per = radii_pos[:, 0]
        r_apo = radii_pos[:, 1]

        eccentricity = (r_apo - r_per) / (r_apo + r_per)

        ep_agama = pot_nbody.potential(np.array(init_cond_lst)[:, 0:3])
        et = ep_agama + ek

        # It has been found that some GCs (e.g. at snap 214) are not bound (et > 0)
        # This has resulted in failed normalisation by circular velocities
        bound_flag = (et <= 0).astype(int)

        ##### add circular normalised data #####
        l_xyz = snap_data["l.xyz"][()]
        lz = l_xyz[:, 2]

        # get max eigenvalue
        # tideig1 = snap_data["tideig_1"][()]
        # tideig2 = snap_data["tideig_2"][()]
        # tideig3 = snap_data["tideig_3"][()]

        # tideig_m = np.max(np.abs((tideig1, tideig2, tideig3)), axis=0)

        if len(et) == 1:
            # pot.Rcirc it didn't like handling arrays of 1
            et = et[0]

        r_circs = pot_nbody.Rcirc(E=et)
        xyz = np.column_stack((r_circs, r_circs * 0, r_circs * 0))
        v_circs = np.sqrt(-r_circs * pot_nbody.force(xyz)[:, 0])
        vel = np.column_stack((v_circs * 0, v_circs, v_circs * 0))
        init_conds = np.concatenate((xyz, vel), axis=1)
        lz_circ = af(init_conds)[:, 2]

        E_0 = pot_nbody.potential((0, 0, 0))

        lz_norm = lz / np.array(lz_circ)
        et_norm = et / np.abs(E_0)

        kin_dict = {
            "r_per": r_per,
            "r_apo": r_apo,
            "ep_agama": ep_agama,
            "et": et,
            "j.cyl": j_cyl,
            "inc": inclination,
            "ecc": eccentricity,
            "lz_norm": lz_norm,
            "et_norm": et_norm,
            "bound_flag": bound_flag,
            # "tideig_m": tideig_m,
        }

        it_dict[it_id] = kin_dict
    data_dict[snap_id] = it_dict

    proc_data.close()

    return data_dict
