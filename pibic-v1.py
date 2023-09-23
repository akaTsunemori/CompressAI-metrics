import matplotlib.pyplot as plt
import piq
import torch
import subprocess
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
    target_img_path = 'pibic-tests/original.png'
    target_img = torch.tensor(imread(target_img_path)).permute(2, 0, 1)[None, ...] / 255.
    for metric in metrics:
        plt.figure(figsize=(12.8, 7.2), dpi=300)
        for model in models:
            qualities = models[model]
            result_x = []
            result_y = []
            for quality in qualities:
                encode_cmd = f'python3 examples/codec.py encode pibic-tests/original.png --model {model} -q {quality} -o pibic-tests/img/{model}_q{quality}'
                decode_cmd = f'python3 examples/codec.py decode pibic-tests/img/{model}_q{quality} -o pibic-tests/img/{model}_q{quality}.png'
                encode_subprocess = subprocess.run(encode_cmd, shell=True, capture_output=True)
                subprocess.run(decode_cmd, shell=True, capture_output=False)
                img_bpp = float(encode_subprocess.stdout.decode('utf-8').split()[0])
                img_path = f'pibic-tests/img/{model}_q{quality}.png'
                input_img = torch.tensor(imread(img_path)).permute(2, 0, 1)[None, ...] / 255.
                img_metric = metric(input_img, target_img).item()
                result_x.append(img_bpp)
                result_y.append(img_metric)
            plt.plot(result_x, result_y, label=f'{model}', marker='o')
        plt.grid(visible=True)
        plt.xlabel('Bit-rate [bpp]')
        plt.ylabel(metrics[metric])
        plt.legend()
        plt.savefig(f'pibic-tests/results/{metrics[metric]}.png')


if __name__ == '__main__':
    main()
