import agama
import h5py
import numpy as np
import utilities as ut
from gc_utils import iteration_name, snapshot_name  # type: ignore


def get_kinematics(
    part, sim: str, it_lst: list[int], snapshot: int, sim_dir: str, data_dir: str, host_index: int = 0
):
    snap_id = snapshot_name(snapshot)

    proc_file = sim_dir + sim + "/" + sim + "_processed.hdf5"
    potential_file = data_dir + "potentials/" + sim + "/snap_%d/combined_snap_%d.ini" % (snapshot, snapshot)

    proc_data = h5py.File(proc_file, "a")  # open processed data file
    pot_nbody = agama.Potential(potential_file)
    af = agama.ActionFinder(pot_nbody, interp=False)

    for it in it_lst:
        it_id = iteration_name(it)

        gc_id_snap = proc_data[it_id]["snapshots"][snap_id]["gc_id"]

        if len(gc_id_snap) is None:
            continue

        ptype_byte_snap = proc_data[it_id]["snapshots"][snap_id]["ptype"]
        ptype_snap = [ptype.decode("utf-8") for ptype in ptype_byte_snap]

        host_name = ut.catalog.get_host_name(host_index)

        x_lst = []
        y_lst = []
        z_lst = []
        vx_lst = []
        vy_lst = []
        vz_lst = []

        r_cyl_lst = []
        phi_cyl_lst = []
        vr_cyl_lst = []
        vphi_cyl_lst = []

        init_cond_lst = []

        r_lst = []

        ep_fire_lst = []
        ek_lst = []

        lx_lst = []
        ly_lst = []
        lz_lst = []

        for gc, ptype in zip(gc_id_snap, ptype_snap):
            idx = np.where(part[ptype]["id"] == gc)[0][0]
            pos_xyz = part[ptype].prop(f"{host_name}.distance.principal", idx)
            vel_xyz = part[ptype].prop(f"{host_name}.velocity.principal", idx)

            pos_cyl = part[ptype].prop(f"{host_name}.distance.principal.cylindrical", idx)
            vel_cyl = part[ptype].prop(f"{host_name}.velocity.principal.cylindrical", idx)

            init_cond = np.concatenate((pos_xyz, vel_xyz))

            ep_fir = part[ptype]["potential"][idx]
            ek = 0.5 * np.linalg.norm(vel_xyz) ** 2

            x, y, z = pos_xyz
            vx, vy, vz = vel_xyz

            r_cyl, phi_cyl, _ = pos_cyl
            vr_cyl, vphi_cyl, _ = vel_cyl

            r = np.linalg.norm(pos_xyz)

            lx = y * vz - z * vy
            ly = z * vx - x * vz
            lz = x * vy - y * vx

            x_lst.append(x)
            y_lst.append(y)
            z_lst.append(z)
            vx_lst.append(vx)
            vy_lst.append(vy)
            vz_lst.append(vz)

            r_cyl_lst.append(r_cyl)
            phi_cyl_lst.append(phi_cyl)
            vr_cyl_lst.append(vr_cyl)
            vphi_cyl_lst.append(vphi_cyl)

            r_lst.append(r)

            ep_fire_lst.append(ep_fir)
            ek_lst.append(ek)

            lx_lst.append(lx)
            ly_lst.append(ly)
            lz_lst.append(lz)

            init_cond_lst.append(init_cond)

        ioms = af(init_cond_lst)
        jr_lst = ioms[:, 0]
        jz_lst = ioms[:, 1]
        jphi_lst = ioms[:, 2]

        radii_pos = pot_nbody.Rperiapo(init_cond_lst)
        r_per_lst = radii_pos[:, 0]
        r_apo_lst = radii_pos[:, 1]

        ep_agama_lst = pot_nbody.potential(np.array(init_cond_lst)[:, 0:3])
        et_lst = ep_agama_lst + ek_lst

        kin_dict = {
            "x": x_lst,
            "y": y_lst,
            "z": z_lst,
            "vx": vx_lst,
            "vy": vy_lst,
            "vz": vz_lst,
            "r_cyl": r_cyl_lst,
            "phi_cyl": phi_cyl_lst,
            "vr_cyl": vr_cyl_lst,
            "vphi_cyl": vphi_cyl_lst,
            "r": r_lst,
            "r_peri": r_per_lst,
            "r_apoo": r_apo_lst,
            "ep_fire": ep_fire_lst,
            "ep_agama": ep_agama_lst,
            "ek": ek_lst,
            "et": et_lst,
            "lx": lx_lst,
            "ly": ly_lst,
            "lz": lz_lst,
            "jr": jr_lst,
            "jz": jz_lst,
            "jphi": jphi_lst,
        }

        # with h5py.File(data_file, "a") as hdf:
        if it_id in proc_data.keys():
            grouping = proc_data[it_id]
        else:
            grouping = proc_data.create_group(it_id)
        if "snapshots" in grouping.keys():
            kinematics = grouping["snapshots"]
        else:
            kinematics = grouping.create_group("snapshots")
        if snap_id in kinematics.keys():
            snap_group = kinematics[snap_id]
        else:
            snap_group = kinematics.create_group(snap_id)
        for key in kin_dict.keys():
            if key in snap_group.keys():
                del snap_group[key]

            snap_group.create_dataset(key, data=kin_dict[key])

    proc_data.close()
