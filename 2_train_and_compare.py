
from utils.codec import main as codec
from utils.train import main as train

import matplotlib.pyplot as plt
import piq
import torch
from collections import defaultdict
from os import rename, mkdir
from os.path import getsize, exists
from PIL import Image
from skimage.io import imread


def calculate_metrics(target_img, w, h, model, quality, metrics, results):
    compressed_size = getsize(f'results/img/{model}_q{quality}')
    img_bpp = (compressed_size * 8) / (w * h)
    img_path = f'results/img/{model}_q{quality}.png'
    input_img = torch.tensor(imread(img_path)).permute(2, 0, 1)[None, ...] / 255.
    for metric in metrics:
        img_metric = metric(input_img, target_img).item()
        results[metrics[metric]][model]['x'].append(img_bpp)
        results[metrics[metric]][model]['y'].append(img_metric)


def main():
    models = { # Model: Quality range
        'bmshj2018-factorized': range(1, 9),
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
    train_dataset_path = './dataset'
    target_img_path = './static/original.png'
    w, h = Image.open(target_img_path).size
    target_img = torch.tensor(imread(target_img_path)).permute(2, 0, 1)[None, ...] / 255.
    # results[metric][model][axis] = list of bpp[x]/metric[y] values
    results_pretrained = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    results_custom     = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if not exists('results/img'):
        mkdir('results/img')
    if not exists('results/train_and_compare'):
        mkdir('results/train_and_compare')
    for model in models:
        qualities = models[model]
        for quality in qualities:
            encoded_output = f'results/img/{model}_q{quality}'
            decoded_output = f'results/img/{model}_q{quality}.png'
            # Train model
            trained_model = f'{model}_{quality}_best_loss.pth.tar'
            checkpoint    = f'{model}_{quality}_checkpoint.pth.tar'
            if not exists(trained_model):
                train([
                    '--model', model,
                    '--dataset', train_dataset_path,
                    '--epochs', '100',
                    '--quality', str(quality),
                    '--cuda', '--save'])
                rename('checkpoint_best_loss.pth.tar', trained_model)
                rename('checkpoint.pth.tar', checkpoint)
            # Use the pretrained model
            codec([
                'encode', target_img_path,
                '--output', encoded_output,
                '--model', model,
                '--quality', str(quality),
                '--pretrained', '1',
            ])
            codec([
                'decode', encoded_output,
                '--output', decoded_output,
                '--pretrained', '1',
            ])
            calculate_metrics(target_img, w, h, model, quality, metrics, results_pretrained)
            # Use the custom-trained model
            codec([
                'encode', target_img_path,
                '--output', encoded_output,
                '--model', model,
                '--quality', str(quality),
                '--pretrained', '0',
                '--state_dict', trained_model
            ])
            codec([
                'decode', encoded_output,
                '--output', decoded_output,
                '--pretrained', '0',
                '--state_dict', trained_model,
            ])
            calculate_metrics(target_img, w, h, model, quality, metrics, results_custom)
    results = results_pretrained
    for metric in results:
        for model in results[metric]:
            plt.figure(figsize=(12.8, 7.2), dpi=300)
            x, y = results_pretrained[metric][model]['x'], results_pretrained[metric][model]['y']
            plt.plot(x, y, label=f'{model} pretrained', marker='o')
            x, y = results_custom[metric][model]['x'], results_custom[metric][model]['y']
            plt.plot(x, y, label=f'{model} custom-trained', marker='o')
            plt.grid(visible=True)
            plt.xlabel('Bit-rate [bpp]')
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(f'results/train_and_compare/{model}_{metric}.png')


if __name__ == '__main__':
    main()
