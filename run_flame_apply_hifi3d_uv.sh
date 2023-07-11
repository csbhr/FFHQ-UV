#!/bin/bash
set -e


######################### Configuration #########################
# flame_mesh_path: the path of the mesh of the head which is with FLAME topology.
# uvmap_name: the filename of the UV-texture map which is with HIFI3D++ topology. It should be in the same directory with mesh.
#################################################################
flame_mesh_path=../examples/flame_apply_hifi3d_uv_examples/flame_head.obj
uvmap_name=hifi3d_tex_uv.png


#################### Run Add Eyeball Script #####################
# Read the head mesh of ${flame_mesh_path}
# Save the processed head mesh in the directory of ${mesh_path}, and link it to the UV-map ${uvmap_name}.
#################################################################
cd ./FLAME_Apply_HIFI3D_UV
python run_flame_apply_hifi3d_uv.py --flame_mesh_path ${flame_mesh_path} --uvmap_name ${uvmap_name}
