import trimesh, time
import igl
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from mesh import TriMesh

def random_rot(max_degrees=20):
    rand_angle = np.random.rand(3) * max_degrees
    # rand_angle[:2] = 0
    rot = Rotation.from_euler('xyz', rand_angle, degrees=True).as_matrix()
    return rot

def random_translation(_min, _max):
    return np.random.rand(3) * (_max-_min) + _min

def trans(points:np.ndarray, rot=None, trans=None):
    if rot is None:
        rot = np.eye(3)
    if trans is None:
        trans = np.zeros(3)

    c = points.mean(0)
    return (points-c) @ rot + trans + c

def random_trans(points:np.ndarray):
    rot = random_rot(45)
    t = random_translation(-1, 1)
    return trans(points, rot, t)

if __name__ == '__main__':

    with open('gum_data', 'rb') as f:
        vs, fs, handle_dict = pickle.load(f)
    vs = np.array(vs).astype('f4')
    fs = np.array(fs).astype('i4')

    handle_idx = np.concatenate(list(handle_dict.values()))

    b = handle_idx.astype('i4')
    arap = igl.ARAP(vs, fs, 3, b)

    idx = handle_dict[36]
    while True:
        bc = vs.copy()
        bc[idx] = random_trans(bc[idx])

        # b = np.unique(b)
        bc = bc[b].copy()
        bc = bc.astype('f4')

        t1 = time.time()
        # vn2 = arap2.solve(bc, vs)
        vn = arap.solve(bc, vs)
        print(time.time()-t1)
        TriMesh(vn, fs).visualize_switch(TriMesh(vs, fs))
