import os
import argparse
import shutil


def copy_from_fitted_results(ori_path, dst_path, ori_name, dst_name_template):
    os.makedirs(dst_path, exist_ok=True)
    fnames = sorted(os.listdir(ori_path))
    for fn in fnames:
        if not os.path.isdir(os.path.join(ori_path, fn)):
            continue
        shutil.copy2(
            src=os.path.join(ori_path, fn, ori_name),
            dst=os.path.join(dst_path, dst_name_template.format(fn))
        )
        print('Copy', ori_name, dst_name_template.format(fn))


if __name__ == '__main__':
    '''Usage
    cd ./RGB_Fitting
    python step3_copy_fitted_mesh.py \
        --fitted_results_dir ../examples/fitting_examples/outputs \
        --mesh_dir ../examples/fitting_examples/outputs-fitted_mesh \
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fitted_results_dir',
                        type=str,
                        default='../data/fitting_examples/outputs',
                        help='directory of the fitted results')
    parser.add_argument('--mesh_dir',
                        type=str,
                        default='../data/fitting_examples/outputs-fitted_mesh',
                        help='directory of copied meshes')
    args = parser.parse_args()

    copy_from_fitted_results(
        ori_path=args.fitted_results_dir,
        dst_path=args.mesh_dir,
        ori_name='stage3_mesh_exp.obj',
        dst_name_template='{}.obj'
    )

    os.makedirs(args.mesh_dir, exist_ok=True)
    fnames = sorted(os.listdir(args.fitted_results_dir))
    for fn in fnames:
        if not os.path.isdir(os.path.join(args.fitted_results_dir, fn)):
            continue
        shutil.copy2(
            src=os.path.join(args.fitted_results_dir, fn, 'stage3_mesh_exp.obj'),
            dst=os.path.join(args.mesh_dir, '{}.obj'.format(fn))
        )
        print(f'Copy {fn}/stage3_mesh_exp.obj to {fn}.obj.')




