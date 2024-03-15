import argparse
import importlib

import json
import numpy as np
import os
import torch
from tqdm import tqdm

from data_loader.general_data_loader import GeneralDataset
from parse_config import ConfigParser
import yaml
from matplotlib import pyplot as plt

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    print(f'==============Configuration==============')
    print(f'{json.dumps(config._config, indent=4)}')
    print(f'==============End Configuration==============')

    logger = config.get_logger('test')

    config['data_loader'] = config['test_data_loader']
    # setup data_loader instances
    my_transform = config.init_transform(training=False)
    data_loader_all_hospitals = {
        'A': config.init_obj('data_loader', transforms=my_transform, training=True),
        'B': config.init_obj('data_loader', transforms=my_transform, training=False)
    }

    # build model architecture
    model = config.init_obj('model')
    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_ftn('loss')
    metric_fns = [getattr(importlib.import_module(f"metric.{met}"), met) for met in config['metric']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    assert config.resume is not None, "In the test script, -r must be assigned to reload well-trained model."
    # remove "module." in case the model was trained on >1 GPUs
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    # force it to use 1 gpu to not muck around with checkpoint:
    config['n_gpu'] = 1
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    save_dir = '{:s}/test_results'.format(str(config.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save test results at {save_dir}")

    viz_data = False # make sure that the data looks reasonable

    with torch.no_grad():
        for hosp, data_loader in data_loader_all_hospitals.items():
            total_loss = 0.0
            total_metrics = torch.zeros(len(metric_fns))

            for i, (data, target, misc) in enumerate(tqdm(data_loader)):
                # print(f"data: {data.min()} {data.max()} | target: {target.min()} {target.max()}")
                if viz_data:
                    data_np = data.detach().cpu().numpy()
                    data_np = (data_np * 127.5) + 127.5
                    data_np = np.round(data_np).astype(np.uint8)
                    target_np = target.detach().cpu().numpy()

                orig_data, target = data.to(device), target.to(device)
                output = model(orig_data)

                if viz_data:
                    output_np = output.detach().cpu().numpy()
                    for j in range(data.shape[0]):
                        data_slice = data_np[j]
                        target_slice = target_np[j]
                        output_slice = output_np[j]
                        # replace with argmax
                        output_slice = np.argmax(output_slice, axis=0)
                        # show side by side
                        fig, ax = plt.subplots(1, 3)
                        data_slice = data_slice.transpose((1, 2, 0))
                        if data_slice.shape[-1] == 1:
                            data_slice = data_slice[..., 0]
                        ax[0].imshow(data_slice)
                        ax[1].imshow(target_slice, cmap='tab10')
                        ax[2].imshow(output_slice, cmap='tab10')
                        plt.savefig(f"{save_dir}/viz_{hosp}_{i}_{j}.png")
                        plt.close()

                loss = criterion[0](output, target, misc)
                if len(criterion) > 1:
                    for idx in range(1, len(criterion)):
                        loss += criterion[idx](output, target, misc)

                batch_size = output.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    misc['input'] = orig_data
                    total_metrics[i] += metric(output, target, misc) * batch_size #TODO need batch_size=1 for this to make sense

            n_samples = len(data_loader.sampler)

            with open(f"{save_dir}/info_evaluate_{hosp}.yaml", 'w') as file:
                to_save = {
                    'hospital': hosp,
                    'test':
                        {
                            'loss': total_loss / n_samples,
                            **{met.__name__: {'avg': total_metrics[i].item() / n_samples} for i, met in enumerate(metric_fns)}
                        }
                }
                logger.info(to_save)
                yaml.dump(to_save, file)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
