import yaml
from pathlib import Path
from typing import List, Tuple
import numpy as np

def get_avg_dice(paths: List[Path]) -> Tuple[List[float], List[float]]:
    dice_a, dice_b = [], []

    for path in paths:
        yaml_files = path.glob('test_results/info_evaluate_[AB].yaml')

        for file in yaml_files:
            with open(file, 'r') as f:
                content = yaml.safe_load(f)
                if content['hospital'] == 'A':
                    dice_a.append(content['test']['dice']['avg'])
                elif content['hospital'] == 'B':
                    dice_b.append(content['test']['dice']['avg'])

    return dice_a, dice_b


def print_results(paths, setting_name):
    dice_a, dice_b = get_avg_dice(paths)
    dice_a = [x * 100 for x in dice_a]
    dice_b = [x * 100 for x in dice_b]
    print(dice_a, dice_b)
    print(f"{setting_name} | A | {np.mean(dice_a):.1f} ± {np.std(dice_a):.1f}")
    print(f"{setting_name} | B | {np.mean(dice_b):.1f} ± {np.std(dice_b):.1f}")


if __name__ == '__main__':
    base_path = Path('/export/scratch2/aleksand/dsl/segmentation/saved/models')

    # paths = [base_path / 'polyp_unet_0/0313_130201',
    #          base_path / 'polyp_unet_1/0312_174329',
    #          base_path / 'polyp_unet_2/0313_131855',
    #          base_path / 'polyp_unet_3/0313_095650',
    #          base_path / 'polyp_unet_4/0313_125856',
    #          ]
    # print_results(paths, 'Polyp')

    paths = [base_path / 'polyp_unet_0_tuned1/0314_075642',
             base_path / 'polyp_unet_1_tuned1/0314_143815',
             base_path / 'polyp_unet_2_tuned1/0314_143855',
             base_path / 'polyp_unet_3_tuned1/0314_143947',
             base_path / 'polyp_unet_4_tuned1/0314_144021',
             ]
    print_results(paths, 'Polyp tuned')

    # paths = [
    #     base_path / 'qata_unet_0/0314_095906',
    #     base_path / 'qata_unet_1/0310_222657',
    #     base_path / 'qata_unet_2/0311_093215',
    #     base_path / 'qata_unet_3/0311_093310',
    #     base_path / 'qata_unet_3/0311_093409',
    # ]
    # print_results(paths, 'Lung')

    # paths = [
    #     base_path / 'cervix_unet_0/0313_094546',
    #     base_path / 'cervix_unet_1/0314_142924',
    #     base_path / 'cervix_unet_2/0314_100810',
    #     base_path / 'cervix_unet_3/0314_143039',
    #     base_path / 'cervix_unet_4/0314_100849',
    # ]
    # print_results(paths, 'Cervix')