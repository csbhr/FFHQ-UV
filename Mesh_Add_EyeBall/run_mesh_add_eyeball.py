import re
import os
import shutil
import numpy as np
from numpy import linalg as la
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import argparse

eye_half_vtx_idx = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
    89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 243, 244, 245, 246, 247, 248, 249, 250,
    251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273,
    274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296,
    297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319,
    320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342,
    343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365,
    366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388,
    389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411,
    412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434,
    435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457,
    458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
    481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503,
    504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526,
    527, 528, 529, 530
]

eye_contour_vtx_idxL = [
    7196, 7635, 7223, 1250, 7199, 1352, 1883, 8472, 8483, 8466, 8502, 8567, 8477, 16946, 16950, 8492, 8489, 8495, 8491
]
eye_contour_vtx_idxR = [
    3790, 12773, 12357, 3897, 12382, 3795, 4428, 13576, 13586, 13570, 13602, 13659, 13579, 13584, 13573, 13593, 13590,
    13597, 13594
]

eye_bag_vtx_idxL = [8536, 8560, 1883, 8517, 8523]
eye_bag_vtx_idxR = [13640, 13663, 4428, 13618, 13625]

eye_ball_vtx_idxL = [755, 152, 442, 160, 176]
eye_ball_vtx_idxR = [755, 136, 458, 160, 208]


def obj_read_quad_tri(file_path):
    vertices = []  # v
    vertices_texture = []  # vt
    vertices_normal = []  # vn

    # notice that faces are converted to triangular faces
    face_v = []  # f 1 2 3
    face_vt = []  # f 1/1 2/2 3/3
    face_vn = []  # f 1/1/1 2/2/2 3/3/3

    tri_v = []

    lines = open(file_path, 'r').readlines()
    for line in lines:
        line = re.sub(' +', ' ', line)
        if line.startswith('v '):
            toks = line.strip().split(' ')[1:]
            try:
                vertices.append([float(toks[0]), float(toks[1]), float(toks[2])])
            except Exception:
                print(toks)
        elif line.startswith('vt '):
            toks = line.strip().split(' ')[1:]
            vertices_texture.append([float(toks[0]), float(toks[1])])
        elif line.startswith('vn '):
            toks = line.strip().split(' ')[1:]
            vertices_normal.append([float(toks[0]), float(toks[1]), float(toks[2])])
        elif line.startswith('f '):
            toks = line.strip().split(' ')[1:]
            if len(toks) == 3:  # triangular faces
                faces1 = toks[0].split('/')
                faces2 = toks[1].split('/')
                faces3 = toks[2].split('/')

                face_v.append(np.array([faces1[0], faces2[0], faces3[0]], np.int32) - 1)
                tri_v.append(np.array([faces1[0], faces2[0], faces3[0]], np.int32) - 1)
                if len(faces1) >= 2:
                    face_vt.append(np.array([faces1[1], faces2[1], faces3[1]], np.int32) - 1)
                if len(faces1) >= 3:
                    if len(faces1[2]) == 0:
                        continue
                    face_vn.append(np.array([faces1[2], faces2[2], faces3[2]], np.int32) - 1)

            if len(toks) == 4:  # quad faces
                faces1 = toks[0].split('/')
                faces2 = toks[1].split('/')
                faces3 = toks[2].split('/')
                faces4 = toks[3].split('/')

                face_v.append(np.array([faces1[0], faces2[0], faces3[0], faces4[0]], np.int32) - 1)
                tri_v.append(np.array([faces1[0], faces2[0], faces3[0]], np.int32) - 1)
                tri_v.append(np.array([faces1[0], faces3[0], faces4[0]], np.int32) - 1)
                if len(faces1) >= 2:
                    face_vt.append(np.array([faces1[1], faces2[1], faces3[1], faces4[1]], np.int32) - 1)
                if len(faces1) >= 3:
                    if len(faces1[2]) == 0:
                        continue
                    face_vn.append(np.array([faces1[2], faces2[2], faces3[2], faces4[2]], np.int32) - 1)

    results = {}
    results['v'] = np.array(vertices, np.float32)
    if len(vertices_texture) > 0:
        results['vt'] = np.array(vertices_texture, np.float32)
    if len(vertices_normal) > 0:
        results['vn'] = np.array(vertices_normal, np.float32)

    if len(face_v) > 0:
        results['fv'] = face_v
        results['tri_v'] = np.array(tri_v)
    if len(face_vt) > 0:
        results['fvt'] = face_vt
    if len(face_vn) > 0:
        results['fvn'] = face_vn

    return results


def obj_write_quad_tri(filename, mtl_name, v, f, vt=None, fvt=None, vn=None, fvn=None):

    with open(filename, 'w') as fp:

        fp.write('mtllib ')
        #fp.write('test.mtl')
        fp.write(mtl_name)
        fp.write('\n')

        for x, y, z in v:
            fp.write('v %f %f %f\n' % (x, y, z))

        if vt is not None:
            for u, v in vt:
                fp.write('vt %f %f\n' % (u, v))

        if vn is not None:
            for x, y, z in vn:
                fp.write('vn %f %f %f\n' % (x, y, z))

        # fp.write('usemtl blinn1SG\n')

        if fvt is None and fvn is None:  # f only
            for v_list in f:
                v_list = v_list + 1
                if len(v_list) == 3:
                    v1, v2, v3 = v_list
                    fp.write('f %d %d %d\n' % (v1, v2, v3))
                else:
                    v1, v2, v3, v4 = v_list
                    fp.write('f %d %d %d %d\n' % (v1, v2, v3, v4))

        elif fvn is None:
            for v_list, vt_list in zip(f, fvt):
                v_list = v_list + 1
                vt_list = vt_list + 1
                if len(v_list) == 3:
                    v1, v2, v3 = v_list
                    t1, t2, t3 = vt_list
                    fp.write('f %d/%d %d/%d %d/%d\n' % (v1, t1, v2, t2, v3, t3))
                else:
                    v1, v2, v3, v4 = v_list
                    t1, t2, t3, t4 = vt_list
                    fp.write('f %d/%d %d/%d %d/%d %d/%d\n' % (v1, t1, v2, t2, v3, t3, v4, t4))

        else:
            for v_list, vt_list, vn_list in zip(f, fvt, fvn):
                v_list = v_list + 1
                vt_list = vt_list + 1
                vn_list = vn_list + 1
                if len(v_list) == 3:
                    v1, v2, v3 = v_list
                    t1, t2, t3 = vt_list
                    n1, n2, n3 = vn_list
                    fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (v1, t1, n1, v2, t2, n2, v3, t3, n3))
                else:
                    v1, v2, v3, v4 = v_list
                    t1, t2, t3, t4 = vt_list
                    n1, n2, n3, n4 = vn_list
                    fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n' %
                             (v1, t1, n1, v2, t2, n2, v3, t3, n3, v4, t4, n4))


def get_ver_norm_np(vtx, tri_list):

    ver_norm = np.zeros_like(vtx)  # N * 3

    v1 = vtx[tri_list[:, 0]]  # NF * 3
    v2 = vtx[tri_list[:, 1]]  # NF * 3
    v3 = vtx[tri_list[:, 2]]  # NF * 3

    l1 = v2 - v3  # NF * 3
    l2 = v1 - v3  # NF * 3

    face_norm = np.cross(l1, l2)  # NF * 3

    for i in range(tri_list.shape[0]):
        tri = tri_list[i]
        ver_norm[tri[0]] = ver_norm[tri[0]] + face_norm[i]
        ver_norm[tri[1]] = ver_norm[tri[1]] + face_norm[i]
        ver_norm[tri[2]] = ver_norm[tri[2]] + face_norm[i]

    ver_norm = ver_norm / np.linalg.norm(ver_norm, axis=1, keepdims=True)  # N * 1

    return ver_norm


def fit_icp_scale_RT(source, target, input_vtx):
    npoint = source.shape[0]
    means = np.mean(source, axis=0)
    meant = np.mean(target, axis=0)

    s1 = source - np.tile(means, (npoint, 1))
    t1 = target - np.tile(meant, (npoint, 1))

    W = t1.transpose().dot(s1)
    U, sigma, VT = la.svd(W)
    rotation = U.dot(VT)

    scale = sum(sum(np.abs(t1))) / sum(sum(abs(rotation.dot(s1.transpose()))))
    translation_icp = target - scale * rotation.dot(source.transpose()).transpose()
    translation_icp = np.mean(translation_icp, axis=0)

    translation = np.zeros(3)
    translation[0:2] = translation_icp[0:2]

    translation[2] = translation_icp[2]
    trans = np.zeros((3, 4))
    trans[:, 0:3] = scale * rotation[:, 0:3]
    trans[:, 3] = translation[:]

    ONE = np.ones([input_vtx.shape[0], 1])
    input_vtx = np.concatenate((input_vtx, ONE), axis=1)
    input_vtx = np.transpose(input_vtx, (1, 0))
    input_vtx = trans.dot(input_vtx)
    input_vtx = np.transpose(input_vtx, (1, 0))

    return trans, input_vtx


class NNSearch(object):

    def __init__(self, ver_dst, n=100):  # target vertices
        self.kd_tree = scipy.spatial.KDTree(ver_dst.T, n)

    def find_nearest_neighbors(self, ver_src):
        # kd-tree to find NN
        N_src = ver_src.shape[1]
        nn_distances, nn_indices = self.kd_tree.query(ver_src.T, 3, p=2)
        nn_indices = np.array(nn_indices, np.int32)
        nn_distances = np.array(nn_distances, np.float32)
        return nn_indices, nn_distances


def opt_eye_position(nn_search, eye_contour, ball_vtx, ball_mesh):
    dis_last = 10.0
    dis_h_count = 0
    for i in range(100):

        nn_indices, nn_distances = nn_search.find_nearest_neighbors(eye_contour.T)

        dis = np.sum(nn_distances)
        res = (eye_contour - ball_vtx[nn_indices[:, 0]])
        dis_height = np.sum(res[0:6, 1])
        D_min = np.min(res[0:6, 2])

        if ((dis_last - dis) < 0.001) or (dis > dis_last) or (D_min < -0.0025):  #0.001
            break
        else:
            eye_contour[:, 2] = eye_contour[:, 2] - 0.0005

            dis_last = dis

            if dis_height > 0.0001:
                eye_contour[:, 1] = eye_contour[:, 1] - 0.0002
                dis_h_count = dis_h_count + 1

    ball_mesh['v'][:, 2] = ball_mesh['v'][:, 2] + 0.0005 * (i + 2)
    ball_mesh['v'][:, 1] = ball_mesh['v'][:, 1] + 0.0002 * (dis_h_count) - D_min * 1.5

    return ball_mesh


def add_eye_ball(head_vtx, L_ball_mesh, R_ball_mesh):

    _, L_ball_mesh['v'] = fit_icp_scale_RT(L_ball_mesh['v'][eye_ball_vtx_idxL], head_vtx[eye_bag_vtx_idxL],
                                           L_ball_mesh['v'])
    _, R_ball_mesh['v'] = fit_icp_scale_RT(R_ball_mesh['v'][eye_ball_vtx_idxR], head_vtx[eye_bag_vtx_idxR],
                                           R_ball_mesh['v'])

    L_ball_vtx = L_ball_mesh['v'][eye_half_vtx_idx]
    R_ball_vtx = R_ball_mesh['v'][eye_half_vtx_idx]
    L_nn_search = NNSearch(L_ball_vtx.T)
    R_nn_search = NNSearch(R_ball_vtx.T)

    L_eye_contour = head_vtx[eye_contour_vtx_idxL]
    R_eye_contour = head_vtx[eye_contour_vtx_idxR]
    '''L eye'''
    L_ball_mesh = opt_eye_position(L_nn_search, L_eye_contour, L_ball_vtx, L_ball_mesh)
    '''R eye'''
    R_ball_mesh = opt_eye_position(R_nn_search, R_eye_contour, R_ball_vtx, R_ball_mesh)

    return L_ball_mesh, R_ball_mesh


if __name__ == '__main__':
    '''Usage
    cd ./Mesh_Add_EyeBall
    python run_mesh_add_eyeball.py --mesh_path ../examples/mesh_add_eyeball_examples/mesh_head.obj
    '''

    parser = argparse.ArgumentParser(description="mesh_add_eyeball")
    parser.add_argument(
        "--mesh_path",
        type=str,
        required=True,
        help="The path of the mesh of the head which is without eyeballs.",
    )
    args = parser.parse_args()

    save_dir = os.path.dirname(args.mesh_path)
    L_ball_mesh_path = './eye_ball_assets/eyeLeft_mesh_move.obj'
    R_ball_mesh_path = './eye_ball_assets/eyeRight_mesh_move.obj'
    ball_tex_path = './eye_ball_assets/eye_ball_tex.png'
    ball_mtl_path = './eye_ball_assets/eye_ball_tex.mtl'
    
    head_mesh = obj_read_quad_tri(args.mesh_path)
    L_ball_mesh = obj_read_quad_tri(L_ball_mesh_path)
    R_ball_mesh = obj_read_quad_tri(R_ball_mesh_path)

    L_ball_mesh, R_ball_mesh = add_eye_ball(head_mesh['v'], L_ball_mesh, R_ball_mesh)
    L_vn = get_ver_norm_np(L_ball_mesh['v'], L_ball_mesh['tri_v']) * -1.0
    R_vn = get_ver_norm_np(R_ball_mesh['v'], R_ball_mesh['tri_v']) * -1.0

    shutil.copy2(src=ball_tex_path, dst=os.path.join(save_dir, 'eye_ball_tex.png'))
    shutil.copy2(src=ball_mtl_path, dst=os.path.join(save_dir, 'eye_ball_tex.mtl'))
    obj_write_quad_tri(os.path.join(save_dir, 'L_ball.obj'),
                        mtl_name='eye_ball_tex.mtl',
                        v=L_ball_mesh['v'],
                        f=L_ball_mesh['fv'],
                        vt=L_ball_mesh['vt'],
                        fvt=L_ball_mesh['fvt'],
                        vn=L_vn,
                        fvn=L_ball_mesh['fv'])
    obj_write_quad_tri(os.path.join(save_dir, 'R_ball.obj'),
                        mtl_name='eye_ball_tex.mtl',
                        v=R_ball_mesh['v'],
                        f=R_ball_mesh['fv'],
                        vt=R_ball_mesh['vt'],
                        fvt=R_ball_mesh['fvt'],
                        vn=R_vn,
                        fvn=R_ball_mesh['fv'])
    