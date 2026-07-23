
import numpy as np

npy_path = r"D:\eplorente\PTv3_PNOA\data\pnoa\processed\muestra_1_677-4615_500x500_urban_train_v2_tile_0_0\segment.npy"
print(npy_path)
npy_file = np.load(npy_path)
print(npy_file)
print(npy_file.shape)