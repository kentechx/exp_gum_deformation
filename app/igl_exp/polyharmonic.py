import numpy as np, glob, os, time
import igl
import trimesh.proximity
import pickle
from mesh import TriMesh

with open('gum_data', 'rb') as f:
    vs, fs, handle_dict = pickle.load(f)
vs = np.array(vs).astype('f4')
fs = np.array(fs).astype('i4')

handle_idx = np.concatenate(list(handle_dict.values()))

b = handle_idx.astype('i4')

bc = vs.copy()
bc[handle_dict[31], 2] -= 10.

# b = np.unique(b)
bc = bc[b].copy()
bc = bc.astype('f4')

for i in range(1,3):
    u = igl.harmonic_weights(vs, fs, b, bc, int(i))
    trimesh.Trimesh(u, fs).show()
