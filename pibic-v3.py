
from examples.codec import main as codec
from examples.train import main as train

import matplotlib.pyplot as plt
import piq
import torch
from collections import defaultdict
from os import system
from os.path import getsize
from PIL import Image
from skimage.io import imread


def calculate_metrics(target_img, w, h, model, quality, metrics, results):
    compressed_size = getsize(f'pibic-tests/img/{model}_q{quality}')
    img_bpp = (compressed_size * 8) / (w * h)
    img_path = f'pibic-tests/img/{model}_q{quality}.png'
    input_img = torch.tensor(imread(img_path)).permute(2, 0, 1)[None, ...] / 255.
    for metric in metrics:
        img_metric = metric(input_img, target_img).item()
        results[metrics[metric]][model]['x'].append(img_bpp)
        results[metrics[metric]][model]['y'].append(img_metric)

def main():
    models = { # Model: Quality range
        'bmshj2018-factorized': range(1, 2),
        # 'bmshj2018-factorized-relu': range(1, 9),
        # 'bmshj2018-hyperprior':      range(1, 9),
        # 'mbt2018-mean':              range(1, 9),
        # 'mbt2018':                   range(1, 9),
        # 'cheng2020-anchor':          range(1, 7),
        # 'cheng2020-attn':            range(1, 7),
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
    train_dataset_path = './pibic-tests/datasets/train'
    target_img_path = 'pibic-tests/original.png'
    w, h = Image.open(target_img_path).size
    target_img = torch.tensor(imread(target_img_path)).permute(2, 0, 1)[None, ...] / 255.
    # results[pretrained=True|False][metric][model][axis] = list of bpp[x]/metric[y] values
    results_pretrained = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    results_custom     = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for model in models:
        qualities = models[model]
        for quality in qualities:
            # model(quality=quality, pretrained=True)
            encoded_output = f'pibic-tests/img/{model}_q{quality}'
            decoded_output = f'pibic-tests/img/{model}_q{quality}.png'
            codec([
                'encode', target_img_path,
                '--output', encoded_output,
                '--model', model,
                '--quality', str(quality),
                '--pretrained', 'True',
            ])
            codec([
                'decode', encoded_output,
                '--output', decoded_output
            ])
            calculate_metrics(target_img, w, h, model, quality, metrics, results_pretrained)

            # # Train model
            # train(['--model', model, '--dataset', train_dataset_path])
            # # to do
            # codec([
            #     'encode', target_img_path,
            #     '--output', encoded_output,
            #     '--model', model,
            #     '--quality', str(quality),
            #     '--pretrained', 'False',
            # ])
            # codec([
            #     'decode', encoded_output,
            #     '--output', decoded_output
            # ])
            # calculate_metrics(target_img, w, h, model, quality, metrics, results, False)


    # for metric in results:
    #     plt.figure(figsize=(12.8, 7.2), dpi=300)
    #     for model in results[metric]:
    #         x, y = results[metric][model]['x'], results[metric][model]['y']
    #         plt.plot(x, y, label=f'{model}', marker='o')
    #     plt.grid(visible=True)
    #     plt.xlabel('Bit-rate [bpp]')
    #     plt.ylabel(metric)
    #     plt.legend()
    #     plt.savefig(f'pibic-tests/results/{metric}.png')


if __name__ == '__main__':
    main()
