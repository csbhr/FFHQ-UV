import os
import re
import numpy as np
import argparse


def read_mesh_obj(file_path):
    vertices = []  # v
    vertices_texture = []  # vt
    vertices_normal = []  # vn

    face_v = []  # f 1 2 3
    face_vt = []  # f 1/1 2/2 3/3
    face_vn = []  # f 1/1/1 2/2/2 3/3/3

    mtl_name = None

    lines = open(file_path, 'r').readlines()
    for line in lines:
        line = re.sub(' +', ' ', line)
        if line.startswith('mtllib '):
            toks = line.strip().split(' ')[1:]
            mtl_name = toks[0]
        elif line.startswith('v '):
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
            if len(toks) == 3:  # tri faces
                faces1 = toks[0].split('/')
                faces2 = toks[1].split('/')
                faces3 = toks[2].split('/')

                face_v.append(np.array([faces1[0], faces2[0], faces3[0]], np.int32) - 1)
                if len(faces1) >= 2 and len(faces1[1]) > 0:
                    face_vt.append(np.array([faces1[1], faces2[1], faces3[1]], np.int32) - 1)
                if len(faces1) >= 3 and len(faces1[2]) > 0:
                    face_vn.append(np.array([faces1[2], faces2[2], faces3[2]], np.int32) - 1)

            if len(toks) == 4:  # quad faces
                faces1 = toks[0].split('/')
                faces2 = toks[1].split('/')
                faces3 = toks[2].split('/')
                faces4 = toks[3].split('/')

                face_v.append(np.array([faces1[0], faces2[0], faces3[0], faces4[0]], np.int32) - 1)
                if len(faces1) >= 2 and len(faces1[1]) > 0:
                    face_vt.append(np.array([faces1[1], faces2[1], faces3[1], faces4[1]], np.int32) - 1)
                if len(faces1) >= 3 and len(faces1[2]) > 0:
                    face_vn.append(np.array([faces1[2], faces2[2], faces3[2], faces4[2]], np.int32) - 1)

    results = {}
    results['v'] = np.array(vertices, np.float32)
    if len(vertices_texture) > 0:
        results['vt'] = np.array(vertices_texture, np.float32)
    if len(vertices_normal) > 0:
        results['vn'] = np.array(vertices_normal, np.float32)

    if len(face_v) > 0:
        results['fv'] = face_v
    if len(face_vt) > 0:
        results['fvt'] = face_vt
    if len(face_vn) > 0:
        results['fvn'] = face_vn
    
    if mtl_name is not None:
        results['mtl_name'] = mtl_name

    return results


def write_mesh_obj(mesh_info, file_path):
    v = mesh_info['v']
    vt = mesh_info['vt'] if 'vt' in mesh_info else None
    vn = mesh_info['vn'] if 'vn' in mesh_info else None
    fv = mesh_info['fv'] if 'fv' in mesh_info else None
    fvt = mesh_info['fvt'] if 'fvt' in mesh_info else None
    fvn = mesh_info['fvn'] if 'fvn' in mesh_info else None
    mtl_name = mesh_info['mtl_name'] if 'mtl_name' in mesh_info else None

    if vt is None:
        rgb_tex = False
    elif vt.shape[1] == 2:
        rgb_tex = False
    elif vt.shape[1] == 3:
        rgb_tex = True

    with open(file_path, 'w') as fp:
        # write mtl info
        if mtl_name is not None:
            fp.write(f'mtllib {mtl_name}\n')

        # write vertices
        if rgb_tex:
            for (x, y, z), (r, g, b) in zip(v, vt):
                fp.write('v %f %f %f %f %f %f\n' % (x, y, z, r, g, b))
        else:
            for x, y, z in v:
                fp.write('v %f %f %f\n' % (x, y, z))

        # write vertex textures (UV coordinates)
        if vt is not None and not rgb_tex:
            for u, v in vt:
                fp.write('vt %f %f\n' % (u, v))

        # write vertex normal
        if vn is not None:
            for x, y, z in vn:
                fp.write('vn %f %f %f\n' % (x, y, z))

        # write faces
        if fv is not None:  # have face
            if rgb_tex or (fvt is None and fvn is None):  # fv only
                for v_list in fv:
                    v_list = v_list + 1
                    if len(v_list) == 3:
                        v1, v2, v3 = v_list
                        fp.write('f %d %d %d\n' % (v1, v2, v3))
                    else:
                        v1, v2, v3, v4 = v_list
                        fp.write('f %d %d %d %d\n' % (v1, v2, v3, v4))
            elif fvn is None:  # fv/fvt
                for v_list, vt_list in zip(fv, fvt):
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
            elif fvt is None:  # fv//fvn
                for v_list, vn_list in zip(fv, fvn):
                    v_list = v_list + 1
                    vn_list = vn_list + 1
                    if len(v_list) == 3:
                        v1, v2, v3 = v_list
                        n1, n2, n3 = vn_list
                        fp.write('f %d//%d %d//%d %d//%d\n' % (v1, n1, v2, n2, v3, n3))
                    else:
                        v1, v2, v3, v4 = v_list
                        n1, n2, n3, n4 = vn_list
                        fp.write('f %d//%d %d//%d %d//%d %d//%d\n' % (v1, n1, v2, n2, v3, n3, v4, n4))
            else:  # fv/fvt/fvn
                for v_list, vt_list, vn_list in zip(fv, fvt, fvn):
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


def write_mtl(mtl_path, uv_path='albedo.png'):
    with open(mtl_path, 'w') as fp:
        fp.write('newmtl blinn1SG\n')
        fp.write('Ka 0.200000 0.200000 0.200000\n')
        fp.write('Kd 1.000000 1.000000 1.000000\n')
        fp.write('Ks 1.000000 1.000000 1.000000\n')
        fp.write('map_Kd ' + uv_path)


if __name__ == '__main__':
    '''Usage
    cd ./FLAME_Apply_HIFI3D_UV
    python run_flame_apply_hifi3d_uv.py --flame_mesh_path ../examples/flame_apply_hifi3d_uv_examples/flame_head.obj --uvmap_name hifi3d_tex_uv.png
    '''

    parser = argparse.ArgumentParser(description="flame_apply_hifi3d_uv")
    parser.add_argument(
        "--flame_mesh_path",
        type=str,
        required=True,
        help="The path of the mesh of the head which is with FLAME topology.",
    )
    parser.add_argument(
        "--uvmap_name",
        type=str,
        default='hifi3d_tex_uv.png',
        help="The filename of the UV-texture map which is with HIFI3D++ topology. It should be in the same directory with mesh.",
    )
    args = parser.parse_args()

    flame_mesh_path = args.flame_mesh_path
    uvmap_name = args.uvmap_name

    refer_mesh_path = './flame2hifi3d_assets/FLAME_w_HIFI3D_UV.obj'
    save_mesh_path = f'{flame_mesh_path[:-4]}_w_HIFI3D_UV.obj'
    save_mtl_path = f'{flame_mesh_path[:-4]}_w_HIFI3D_UV.mtl'

    refer_data = read_mesh_obj(refer_mesh_path)
    flame_data = read_mesh_obj(flame_mesh_path)

    ##### Copy the UV coordinates
    #
    # In 'refer_data', we provide the UV coordinates of our UV-texture map corresponding to the FALME vertices/faces.
    # Using these UV coordinates, the FLAME meshes can look for texture colors from the UV-maps in the FFHQ-UV dataset.
    #
    # Here, we copy: refer_data['vt'] -> flame_data['vt'], refer_data['fvt'] -> flame_data['fvt']
    #
    ##### Notes: Pay attention to the eyeballs
    #
    # FALME mesh contains eyeball vertices, but our UV-texture map does not contain eyeball textures.
    # Therefore, when rendering, one should let the eyeball vertices look for textures from another eyeball texture map.
    #
    # Here, we provide the indices of eyeball vertices: 3931 ~ 5022 (indexed from 0).
    # We also provide an additional eyeball texture map: ./flame2hifi3d_assets/eye_ball_tex.png
    #
    ##### UV coordinates range
    #
    # The range of UV coordinates for the facial area is (0~1, 0~1).
    # The range of UV coordinates for the eyeball area is (2~3, 0~1).
    #
    flame_data['vt'] = refer_data['vt']
    flame_data['fvt'] = refer_data['fvt']
    flame_data['mtl_name'] = os.path.basename(save_mtl_path)
    
    write_mesh_obj(flame_data, save_mesh_path)
    write_mtl(save_mtl_path, uv_path=uvmap_name)

    ### (Optional, only for simple visualization)
    #
    # If you directly visualize the mesh processed above in MeshLab, you will find that the eyeball vertices will directly use the facial textures.
    # This is because two UV-texture maps cannot be specified in one .obj file, so the eyeball vertices directly reuses the given facial UV-texture map.
    # 
    # Simply setting the UV coordinates of the eyeball vertices to (1.0, 1.0) can temporarily masks the eyeball texture in MeshLab.
    # This is only for simple visualization and is not recommended for rendering.
    #
    vt_new = np.reshape(np.array([0.0, 1.0]), [1, 2])
    flame_data['vt'] = np.concatenate([flame_data['vt'], vt_new], axis=0)
    idx_new = flame_data['vt'].shape[0]
    for i in range(len(flame_data['fv'])):
        for j in range(3):
            if 3931 <= flame_data['fv'][i][j] <= 5022:
                flame_data['fvt'][i][j] = idx_new
    write_mesh_obj(flame_data, f'{save_mesh_path[:-4]}_simply_handle_eyeballs.obj')
