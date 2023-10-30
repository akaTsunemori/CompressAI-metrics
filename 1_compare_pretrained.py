import matplotlib.pyplot as plt
import piq
import torch
from collections import defaultdict
from os import system, mkdir
from os.path import getsize, exists
from PIL import Image
from skimage.io import imread


def main():
    models = { # Model: Quality range
        'bmshj2018-factorized':      range(1, 9),
        'bmshj2018-factorized-relu': range(1, 9),
        'bmshj2018-hyperprior':      range(1, 9),
        'mbt2018-mean':              range(1, 9),
        'mbt2018':                   range(1, 9),
        'cheng2020-anchor':          range(1, 7),
        'cheng2020-attn':            range(1, 7),
    }
    metrics = { # Metric callable: Metric name
        piq.psnr: 'PSNR',
        piq.ssim: 'SSIM',
        piq.multi_scale_ssim: 'MS-SSIM',
        piq.information_weighted_ssim: 'IW-SSIM',
        piq.vif_p: 'VIF-P',
        piq.fsim: 'FSIM',
        piq.srsim: 'SR-SIM',
        piq.gmsd: 'GMSD',
        piq.multi_scale_gmsd: 'MS-GMSD',
        piq.vsi: 'VSI',
        piq.dss: 'DSS',
        piq.haarpsi: 'HaarPSI',
        piq.mdsi: 'MDSI',
    }
    target_img_path = 'static/original.png'
    w, h = Image.open(target_img_path).size
    target_img = torch.tensor(imread(target_img_path)).permute(2, 0, 1)[None, ...] / 255.
    # results[metric][model][axis] = list of bpp[x]/metric[y] values
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if not exists('results/img'):
        mkdir('results/img')
    if not exists('results/compare_pretrained'):
        mkdir('results/compare_pretrained')
    for model in models:
        qualities = models[model]
        for quality in qualities:
            encode_cmd = f'python3 utils/codec.py encode {target_img_path} --model {model} -q {quality} -o results/img/{model}_q{quality}'
            decode_cmd = f'python3 utils/codec.py decode results/img/{model}_q{quality} -o results/img/{model}_q{quality}.png'
            system(encode_cmd)
            system(decode_cmd)
            compressed_size = getsize(f'results/img/{model}_q{quality}')
            img_bpp = (compressed_size * 8) / (w * h)
            img_path = f'results/img/{model}_q{quality}.png'
            input_img = torch.tensor(imread(img_path)).permute(2, 0, 1)[None, ...] / 255.
            for metric in metrics:
                img_metric = metric(input_img, target_img).item()
                results[metrics[metric]][model]['x'].append(img_bpp)
                results[metrics[metric]][model]['y'].append(img_metric)
    for metric in results:
        plt.figure(figsize=(12.8, 7.2), dpi=300)
        for model in results[metric]:
            x, y = results[metric][model]['x'], results[metric][model]['y']
            plt.plot(x, y, label=f'{model}', marker='o')
        plt.grid(visible=True)
        plt.xlabel('Bit-rate [bpp]')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f'results/compare_pretrained/{metric}.png')


if __name__ == '__main__':
    main()
