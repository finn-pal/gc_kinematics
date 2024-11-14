import agama
import h5py
import numpy as np
from gc_utils import iteration_name, snapshot_name  # type: ignore


def get_kinematics(
    sim: str,
    it_lst: list[int],
    snapshot: int,
    sim_dir: str,
    data_dir: str,
    data_dict: dict = {},
):
    snap_id = snapshot_name(snapshot)
    print(snap_id)

    proc_file = sim_dir + sim + "/" + sim + "_processed.hdf5"
    potential_file = data_dir + "potentials/" + sim + "/snap_%d/combined_snap_%d.ini" % (snapshot, snapshot)

    proc_data = h5py.File(proc_file, "r")  # open processed data file
    pot_nbody = agama.Potential(potential_file)
    af = agama.ActionFinder(pot_nbody, interp=False)

    it_dict = {}
    for it in it_lst:
        it_id = iteration_name(it)

        snap_data = proc_data[it_id]["snapshots"][snap_id]
        gc_id_snap = snap_data["gc_id"]

        x = np.array(snap_data["x"])
        y = np.array(snap_data["y"])
        z = np.array(snap_data["z"])

        vx = np.array(snap_data["vx"])
        vy = np.array(snap_data["vy"])
        vz = np.array(snap_data["vz"])

        ek = np.array(snap_data["ek"])

        if len(gc_id_snap) is None:
            continue

        init_cond_lst = np.vstack((x, y, z, vx, vy, vz)).T

        ioms = af(init_cond_lst)
        jr = ioms[:, 0]
        jz = ioms[:, 1]
        jphi = ioms[:, 2]

        radii_pos = pot_nbody.Rperiapo(init_cond_lst)
        r_per = radii_pos[:, 0]
        r_apo = radii_pos[:, 1]

        ep_agama = pot_nbody.potential(np.array(init_cond_lst)[:, 0:3])
        et = ep_agama + ek

        kin_dict = {
            "r_peri": r_per,
            "r_apoo": r_apo,
            "ep_agama": ep_agama,
            "et": et,
            "jr": jr,
            "jz": jz,
            "jphi": jphi,
        }

        it_dict[it_id] = kin_dict
    data_dict[snap_id] = it_dict

    return data_dict
