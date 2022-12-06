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


def download_ffhq_uv(dst_dir):
    url_base = 'http://d36kyfewqr49fv.cloudfront.net/ffhq-uv/'
    clip_names = [f'{i:0>3}.zip' for i in range(55)]
    os.makedirs(dst_dir, exist_ok=True)
    for cn in clip_names:
        download_file(url=url_base + cn, dst_path=os.path.join(dst_dir, cn))


def download_ffhq_uv_interpolate(dst_dir):
    url_base = 'http://d36kyfewqr49fv.cloudfront.net/ffhq-uv-interpolate/'
    clip_names = [f'{i:0>3}.zip' for i in range(100)]
    os.makedirs(dst_dir, exist_ok=True)
    for cn in clip_names:
        download_file(url=url_base + cn, dst_path=os.path.join(dst_dir, cn))


def download_ffhq_uv_face_latents(dst_dir):
    url_base = 'http://d36kyfewqr49fv.cloudfront.net/ffhq-uv-face-latents/'
    clip_names = [f'{i:0>3}.zip' for i in range(55)]
    os.makedirs(dst_dir, exist_ok=True)
    for cn in clip_names:
        download_file(url=url_base + cn, dst_path=os.path.join(dst_dir, cn))


def download_ffhq_uv_interpolate_face_latents(dst_dir):
    url_base = 'http://d36kyfewqr49fv.cloudfront.net/ffhq-uv-interpolate-face-latents/'
    clip_names = [f'{i:0>3}.zip' for i in range(100)]
    os.makedirs(dst_dir, exist_ok=True)
    for cn in clip_names:
        download_file(url=url_base + cn, dst_path=os.path.join(dst_dir, cn))


def download_ffhq_uv_face_attributes(dst_dir):
    url = 'http://d36kyfewqr49fv.cloudfront.net/ffhq-uv-face-attributes/attributes_000_054.zip'
    os.makedirs(dst_dir, exist_ok=True)
    download_file(url=url, dst_path=os.path.join(dst_dir, 'attributes_000_054.zip'))


def download_ffhq_uv_interpolate_face_attributes(dst_dir):
    url = 'http://d36kyfewqr49fv.cloudfront.net/ffhq-uv-interpolate-face-attributes/attributes_000_099.zip'
    os.makedirs(dst_dir, exist_ok=True)
    download_file(url=url, dst_path=os.path.join(dst_dir, 'attributes_000_099.zip'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="download")
    parser.add_argument("--dataset", type=str, default='ffhq-uv', help="ffhq-uv/ffhq-uv-interpolate")
    parser.add_argument("--dst_dir", type=str, default='./', help="The save directory.")
    args = parser.parse_args()

    if args.dataset == 'ffhq-uv':
        # download Standard FFHQ-UV dataset
        download_ffhq_uv(dst_dir=os.path.join(args.dst_dir, 'ffhq-uv'))
    elif args.dataset == 'ffhq-uv-interpolate':
        # download FFHQ-UV-Interpolate dataset
        download_ffhq_uv_interpolate(dst_dir=os.path.join(args.dst_dir, 'ffhq-uv-interpolate'))
    elif args.dataset == 'ffhq-uv-face-latents':
        # download face latent codes of Standard FFHQ-UV dataset
        download_ffhq_uv_face_latents(dst_dir=os.path.join(args.dst_dir, 'ffhq-uv-face-latents'))
    elif args.dataset == 'ffhq-uv-interpolate-face-latents':
        # download face latent codes of FFHQ-UV-Interpolate dataset
        download_ffhq_uv_interpolate_face_latents(dst_dir=os.path.join(args.dst_dir, 'ffhq-uv-interpolate-face-latents'))
    elif args.dataset == 'ffhq-uv-face-attributes':
        # download face attributes of Standard FFHQ-UV dataset
        download_ffhq_uv_face_attributes(dst_dir=os.path.join(args.dst_dir, 'ffhq-uv-face-attributes'))
    elif args.dataset == 'ffhq-uv-interpolate-face-attributes':
        # download face attributes of FFHQ-UV-Interpolate dataset
        download_ffhq_uv_interpolate_face_attributes(dst_dir=os.path.join(args.dst_dir, 'ffhq-uv-interpolate-face-attributes'))
    else:
        raise NotImplementedError
