import h5py

data_path = "/nvme0n1/rdt/datas/StackCube-v1/normal_success_demo/episode_1.hdf5"

with h5py.File(data_path, "r") as f:
    print("f:", f)
    print("qpos:", f["/observations/qpos"])
    print("action:", f["/action"])