import os
from tqdm import tqdm
import requests


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

    for cn in clip_names:
        download_file(
            url=url_base + cn,
            dst_path=os.path.join(dst_dir, cn)
        )


if __name__ == '__main__':
    download_ffhq_uv_standard(dst_dir='./')
