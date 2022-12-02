import os
from tqdm import tqdm
import requests
import argparse


def download_file(url, dst_path):
    dst_fn = os.path.basename(dst_path)

    resp = requests.get(url, stream=True)
    file_size = int(resp.headers['content-length'])

    with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, ascii=True, desc=dst_fn) as bar:
        with requests.get(url, stream=True) as r:
            with open(dst_path, 'wb') as fp:
                for chunk in r.iter_content(chunk_size=512):
                    if chunk:
                        fp.write(chunk)
                        bar.update(len(chunk))


def download_ffhq_uv_standard(dst_dir):
    url_base = 'http://d36kyfewqr49fv.cloudfront.net/standard/'
    clip_names = [f'{i:0>3}.zip' for i in range(55)]

    os.makedirs(dst_dir, exist_ok=True)

    for cn in clip_names:
        download_file(url=url_base + cn, dst_path=os.path.join(dst_dir, cn))


def download_ffhq_uv_interpolate(dst_dir):
    url_base = 'http://d36kyfewqr49fv.cloudfront.net/interpolate/'
    clip_names = [f'{i:0>3}.zip' for i in range(100)]

    os.makedirs(dst_dir, exist_ok=True)

    for cn in clip_names:
        download_file(url=url_base + cn, dst_path=os.path.join(dst_dir, cn))


def download_ffhq_uv_standard_face_latent(dst_dir):
    url_base = 'http://d36kyfewqr49fv.cloudfront.net/standard_face_latent/'
    clip_names = [f'{i:0>3}.zip' for i in range(55)]
    attr_names = [f'attributes_{i:0>3}.json' for i in range(55)]
    clip_names = clip_names + attr_names

    os.makedirs(dst_dir, exist_ok=True)

    for cn in clip_names:
        download_file(url=url_base + cn, dst_path=os.path.join(dst_dir, cn))


def download_ffhq_uv_interpolate_face_latent(dst_dir):
    url_base = 'http://d36kyfewqr49fv.cloudfront.net/interpolate_face_latent/'
    clip_names = [f'{i:0>3}.zip' for i in range(100)]
    attr_names = [f'attributes_{i:0>3}.json' for i in range(100)]
    clip_names = clip_names + attr_names

    os.makedirs(dst_dir, exist_ok=True)

    for cn in clip_names:
        download_file(url=url_base + cn, dst_path=os.path.join(dst_dir, cn))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="download")
    parser.add_argument("--dataset", type=str, default='ffhq-uv-standard', help="ffhq-uv-standard/ffhq-uv-interpolate")
    parser.add_argument("--dst_dir", type=str, default='./', help="The save directory.")
    args = parser.parse_args()

    if args.dataset == 'ffhq-uv-standard':
        # download Standard FFHQ-UV dataset
        download_ffhq_uv_standard(dst_dir=os.path.join(args.dst_dir, 'ffhq-uv-standard'))
    elif args.dataset == 'ffhq-uv-interpolate':
        # download FFHQ-UV-Interpolate dataset
        download_ffhq_uv_interpolate(dst_dir=os.path.join(args.dst_dir, 'ffhq-uv-interpolate'))
    elif args.dataset == 'ffhq-uv-standard-face-latent':
        # download face latent codes of Standard FFHQ-UV dataset
        download_ffhq_uv_standard_face_latent(dst_dir=os.path.join(args.dst_dir, 'ffhq-uv-standard-face-latent'))
    elif args.dataset == 'ffhq-uv-interpolate-face-latent':
        # download face latent codes of FFHQ-UV-Interpolate dataset
        download_ffhq_uv_interpolate_face_latent(dst_dir=os.path.join(args.dst_dir, 'ffhq-uv-interpolate-face-latent'))
    else:
        raise NotImplementedError
