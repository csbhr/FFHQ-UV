import os
import time
import numpy as np
from dominate import document, tags
from torch.utils.tensorboard import SummaryWriter

from .data_utils import save_img


class HTML:

    def __init__(self, web_dir, filename='index.html', title='visual'):
        self.web_dir = web_dir
        self.filename = filename
        os.makedirs(web_dir, exist_ok=True)
        self.doc = document(title=title)

    def add_header(self, text):
        with self.doc:
            tags.h3(text)

    def add_line_images(self, impths, txts=None, width=400):
        '''Add images to the HTML file in a line.'''
        with self.doc:
            with tags.table(border=1, style="table-layout: fixed;"):  # Insert a table
                with tags.tr():
                    for i, im in enumerate(impths):
                        with tags.td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with tags.p():
                                tags.img(style="width:%dpx" % width, src=im)
                                if txts is not None:
                                    tags.br()
                                    tags.p(txts[i])

    def save(self):
        '''Save the current content to the HMTL file'''
        html_file = os.path.join(self.web_dir, self.filename)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


class Logger:

    def __init__(self, vis_dir, flag, prefix=None, is_webpage=False, is_tb=False):
        self.vis_dir = vis_dir
        self.is_webpage = is_webpage
        self.is_tb = is_tb

        os.makedirs(vis_dir, exist_ok=True)

        now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        log_file_path = os.path.join(vis_dir, f'log_{flag}_{now_time}.txt')
        open_type = 'a' if os.path.exists(log_file_path) else 'w'
        self.txt_logger = open(log_file_path, open_type)

        self.prefix = prefix

        if is_webpage:
            self.web_logger = HTML(web_dir=vis_dir)

        if is_tb:
            self.tb_logger = SummaryWriter(os.path.join(vis_dir, f'tb_logs_{flag}'))

    def reset_prefix(self, prefix=None):
        self.prefix = prefix

    def write_txt_log(self, mess):
        prefix = '' if self.prefix is None else f'[{self.prefix}] '
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log = f'[{now_time}] {prefix}{mess}'
        print(log)
        self.txt_logger.write(log + '\n')

    def write_disk_images(self, imgs, txts):
        prefix = '' if self.prefix is None else f'{self.prefix}_'
        for im, tx in zip(imgs, txts):
            save_img(im, os.path.join(self.vis_dir, f'{prefix}{tx}.png'))

    def write_web_header(self, mess):
        assert self.is_webpage
        prefix = '' if self.prefix is None else f'{self.prefix}: '
        self.web_logger.add_header(prefix + mess)

    def write_web_images(self, imgs, txts):
        assert self.is_webpage
        prefix = '' if self.prefix is None else f'{self.prefix}_'
        impths = []
        for im, tx in zip(imgs, txts):
            save_img(im, os.path.join(self.vis_dir, f'{prefix}{tx}.png'))
            impths.append(f'{prefix}{tx}.png')
        self.web_logger.add_line_images(impths, txts)

    def write_tb_scalar(self, names, values, iter):
        assert self.is_tb
        prefix = '' if self.prefix is None else f'{self.prefix}/'
        for name, value in zip(names, values):
            self.tb_logger.add_scalar(prefix + name, value, iter)

    def write_tb_images(self, imgs, txts, iter):
        assert self.is_tb
        prefix = '' if self.prefix is None else f'{self.prefix}/'
        for im, tx in zip(imgs, txts):
            im = np.squeeze(np.clip(np.round(im), 0, 255).astype(np.uint8))
            self.tb_logger.add_image(prefix + tx, im, iter, dataformats='HWC')

    def close(self):
        self.txt_logger.close()
        if self.is_webpage:
            self.web_logger.save()
        if self.is_tb:
            self.tb_logger.close()
