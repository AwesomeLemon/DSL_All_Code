import argparse
import importlib

import h5py
import numpy as np
import os
# import plotly.graph_objects as go
import torch
from datetime import datetime
# from plotly.offline import plot
from tqdm import tqdm
# from utils import MetricTracker

# from utils.clean_noise import CleanNoise

# from data_loader.general_data_loader import GeneralDataset
from parse_config import ConfigParser
import metric.hd95_and_sd_all_deepmind as hd_sd_all
import metric.dice_all as dice_all
import PIL.Image as Image

from utils import adjust_dynamic_range, convert_to_uint8

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('test')

    # config['data_loader'] = config['test_data_loader']
    # setup data_loader instances
    my_transform = config.init_transform(training=False)
    data_loader = config.init_obj('test_data_loader', transforms=my_transform, training=True)

    # build model architecture
    model = config.init_obj('model')
    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_ftn('loss')
    metric_fns = [getattr(importlib.import_module(f"metric.{met}"), met) for met in config['metric']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    assert config.resume is not None, "In the test script, -r must be assigned to reload well-trained model."
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    if 'save_mask' not in config.config:
        config.config.save_mask = False
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))


    now = datetime.now()
    # save_dir = '{:s}/test_results'.format(str(config.save_dir))
    save_dir = os.path.join(config['trainer']['save_dir'], "test_results", config['name'], now.strftime("%Y%m%d%H%M%S"))
    imgs_save_dir = os.path.join(config['trainer']['save_dir'], "test_results", config['name'], now.strftime("%Y%m%d%H%M%S"),"imgs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(imgs_save_dir, exist_ok=True)
    print(f"Save test results at {save_dir}")

    dice_list = []
    dice_cat1_list = []
    dice_cat2_list = []
    dice_cat3_list = []
    dice_cat4_list = []
    dice_cat5_list = []
    dice_cat6_list = []
    dice_cat7_list = []

    sd_list = []
    sd_cat1_list = []
    sd_cat2_list = []
    sd_cat3_list = []
    sd_cat4_list = []
    sd_cat5_list = []
    sd_cat6_list = []
    sd_cat7_list = []

    hd_list = []
    hd_cat1_list = []
    hd_cat2_list = []
    hd_cat3_list = []
    hd_cat4_list = []
    hd_cat5_list = []
    hd_cat6_list = []
    hd_cat7_list = []


    # clean_noise = CleanNoise(top_num=1)

    # test_metrics = MetricTracker('loss', *[m.__name__ for m in metric_fns], writer=self.writer)

    pred_list = {}

    with torch.no_grad():
        for i, (data, target, misc) in enumerate(tqdm(data_loader)):
            #for evaluate surface distance only
            misc['spacing_mm'] = (0.8,0.8,0.8)
            orig_data, target = data.to(device), target.to(device)
            output = model(orig_data)

            # output = output.cpu().numpy()
            # # Remove small objects
            # new_output = []
            # for i in range(output.shape[0]):
            #     output_i = output[i]
            #     mask = np.argmax(output_i, axis=0)
            #     clean_mask = clean_noise.clean_small_obj(mask)
            #     output_i[1][clean_mask==0] = (output_i[0].min()-1)
            #     new_output.append(output_i)
            # new_output = torch.tensor(np.stack(new_output, axis=0)).float().cuda()

            new_output = output


            if hasattr(config.config,"visual") and config.config.visual:
                pad_h,pad_w = misc["padding"]
                pad_h = pad_h[0].item()
                pad_w = pad_w[0].item()
                ##################################################
                # Visualize images.
                if i == 0:
                    print(f"only visualize images....")
                hstack = []
                gt = np.squeeze(target.cpu().numpy())*255/8
                gt = np.rint(gt).clip(0, 255).astype(np.uint8)
                pred = np.squeeze(np.argmax(output.cpu().numpy(), axis=1))*255/8
                pred = np.rint(pred).clip(0, 255).astype(np.uint8)
                data = np.squeeze(data.cpu().numpy())
                data = convert_to_uint8(data,(data.min(),data.max()))
                # remove all padding
                if pad_h != 0:
                    data = data[pad_h:-pad_h,pad_w:-pad_w]
                    gt = gt[pad_h:-pad_h,pad_w:-pad_w]
                    pred = pred[pad_h:-pad_h,pad_w:-pad_w]
                hstack.append(data)
                hstack.append(gt)
                hstack.append(pred)
                himg = np.hstack(hstack).astype("uint8")
                id = misc['img_path'][0].split("/")[1]
                Image.fromarray(himg).save(os.path.join(imgs_save_dir,f"{id}.png"))

                ##################################################

            loss = criterion[0](new_output, target, misc)
            if len(criterion) > 1:
                for idx in range(1, len(criterion)):
                    loss += criterion[idx](new_output, target, misc)

            batch_size = new_output.shape[0]
            total_loss += loss.item() * batch_size

            for i in range(new_output.shape[0]):
                output_i = new_output[i]
                target_i = target[i]
                source_i = orig_data[i]
                id = misc['img_path'][i].split("/")[1].split("-")[0]
                if "_" in misc['img_path'][i]:
                    if len(misc['img_path'][i].split("/")[1].split("_")) == 2:
                        id = misc['img_path'][i].split("/")[1].split("_")[0]
                    # the Id pattern is db_id + "_" + case_id + "_" + slice_id
                    else:
                        id = ("_").join(misc['img_path'][i].split("/")[1].split("_")[:2])
                if id not in pred_list:
                    pred_list[id] = [[],[],[], []]

                pred_list[id][0].append(target_i.cpu().numpy())
                pred_list[id][1].append(output_i.cpu().numpy())
                pred_list[id][2].append(source_i.cpu().numpy())
                pred_list[id][3].append(misc['img_path'][i].split("/")[1].split("_")[-1])


                # if np.count_nonzero(target_i.cpu().numpy()) < 10:
                #     # print(np.count_nonzero(target_i.cpu().numpy()))
                #     continue

                # dice_list.append(metric_fns[0](output_i, target_i, misc))
                # sensitivity_list.append(metric_fns[1](output_i, target_i, misc))
                # specificity_list.append(metric_fns[2](output_i, target_i, misc))
                # hausdorff95_list.append(metric_fns[3](output_i, target_i, misc))
                # file_id = misc['img_path'][i].split("/")[1]
                # id_list.append(file_id)
            # print(metric_fns[0](new_output, target, misc))
            # print(np.average(dice_list))

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    # logger.info(log)

    # if "spacing_mm" in misc:
    #     print('spacing_mm', misc['spacing_mm'])
    # else:
    #     print('no spacing_mm')
    misc['spacing_mm'] = (0.8, 0.8, 0.8)

    if not hasattr(config.config,"visual") or not config.config.visual:
        metrics = h5py.File(f'{save_dir}/metrics.h5','w')
        for key in pred_list:
            print(f"stack and calculate 3D performance...({key})")
            case_result = pred_list[key]
            target = np.stack(case_result[0])
            output = np.stack(case_result[1])
            source = np.stack(case_result[2])
            slice_ids = case_result[3]
            # print(output.shape, target.shape)
            sorted_ids = np.argsort(np.array(slice_ids).astype(np.int))
            target = target[sorted_ids]
            output = output[sorted_ids]
            source = source[sorted_ids]

            dices = dice_all.dice_all(output, target, misc)
            dice_list.append(np.mean(dices))
            dice_cat1_list.append(dices[0])
            dice_cat2_list.append(dices[1])
            dice_cat3_list.append(dices[2])
            dice_cat4_list.append(dices[3])
            dice_cat5_list.append(dices[4])
            dice_cat6_list.append(dices[5])
            dice_cat7_list.append(dices[6])

            hds, sds = hd_sd_all.hd95_and_sd_all(output, target, misc)
            sd_list.append(np.mean(sds))
            sd_cat1_list.append(sds[0])
            sd_cat2_list.append(sds[1])
            sd_cat3_list.append(sds[2])
            sd_cat4_list.append(sds[3])
            sd_cat5_list.append(sds[4])
            sd_cat6_list.append(sds[5])
            sd_cat7_list.append(sds[6])

            hd_list.append(np.mean(hds))
            hd_cat1_list.append(hds[0])
            hd_cat2_list.append(hds[1])
            hd_cat3_list.append(hds[2])
            hd_cat4_list.append(hds[3])
            hd_cat5_list.append(hds[4])
            hd_cat6_list.append(hds[5])
            hd_cat7_list.append(hds[6])

            # dice_list.append(metric_fns[0](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            # dice_cat1_list.append(metric_fns[1](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            # dice_cat2_list.append(metric_fns[2](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            # dice_cat3_list.append(metric_fns[3](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            # dice_cat4_list.append(metric_fns[4](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            # dice_cat5_list.append(metric_fns[5](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            # dice_cat6_list.append(metric_fns[6](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            # dice_cat7_list.append(metric_fns[7](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))

            # if len(metric_fns) > 8:
            #     sd_list.append(metric_fns[8](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     sd_cat1_list.append(metric_fns[9](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     sd_cat2_list.append(metric_fns[10](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     sd_cat3_list.append(metric_fns[11](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     sd_cat4_list.append(metric_fns[12](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     sd_cat5_list.append(metric_fns[13](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     sd_cat6_list.append(metric_fns[14](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     sd_cat7_list.append(metric_fns[15](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #
            # if len(metric_fns) > 16:
            #     hd_list.append(metric_fns[16](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     hd_cat1_list.append(metric_fns[17](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     hd_cat2_list.append(metric_fns[18](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     hd_cat3_list.append(metric_fns[19](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     hd_cat4_list.append(metric_fns[20](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     hd_cat5_list.append(metric_fns[21](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     hd_cat6_list.append(metric_fns[22](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))
            #     hd_cat7_list.append(metric_fns[23](torch.tensor(output).cuda(), torch.tensor(target).cuda(), misc))

            if config.config.save_mask:
                metrics.create_dataset(f"{key}/output", data=output, compression="gzip")
                metrics.create_dataset(f"{key}/target", data=target, compression="gzip")
                metrics.create_dataset(f"{key}/source", data=source, compression="gzip")


        metrics.create_dataset(f"dice", data=np.array(dice_list), compression='gzip')
        metrics.create_dataset(f"dice_cat1", data=np.array(dice_cat1_list), compression='gzip')
        metrics.create_dataset(f"dice_cat2", data=np.array(dice_cat2_list), compression='gzip')
        metrics.create_dataset(f"dice_cat3", data=np.array(dice_cat3_list), compression='gzip')
        metrics.create_dataset(f"dice_cat4", data=np.array(dice_cat4_list), compression='gzip')
        metrics.create_dataset(f"dice_cat5", data=np.array(dice_cat5_list), compression='gzip')
        metrics.create_dataset(f"dice_cat6", data=np.array(dice_cat6_list), compression='gzip')
        metrics.create_dataset(f"dice_cat7", data=np.array(dice_cat7_list), compression='gzip')
        if len(metric_fns) > 8:
            metrics.create_dataset(f"surface_distance", data=np.array(sd_list), compression='gzip')
            metrics.create_dataset(f"surface_distance_cat1", data=np.array(sd_cat1_list), compression='gzip')
            metrics.create_dataset(f"surface_distance_cat2", data=np.array(sd_cat2_list), compression='gzip')
            metrics.create_dataset(f"surface_distance_cat3", data=np.array(sd_cat3_list), compression='gzip')
            metrics.create_dataset(f"surface_distance_cat4", data=np.array(sd_cat4_list), compression='gzip')
            metrics.create_dataset(f"surface_distance_cat5", data=np.array(sd_cat5_list), compression='gzip')
            metrics.create_dataset(f"surface_distance_cat6", data=np.array(sd_cat6_list), compression='gzip')
            metrics.create_dataset(f"surface_distance_cat7", data=np.array(sd_cat7_list), compression='gzip')

        if len(metric_fns) > 16:
            metrics.create_dataset(f"hausdorff_distance", data=np.array(hd_list), compression='gzip')
            metrics.create_dataset(f"hausdorff_distance_cat1", data=np.array(hd_cat1_list), compression='gzip')
            metrics.create_dataset(f"hausdorff_distance_cat2", data=np.array(hd_cat2_list), compression='gzip')
            metrics.create_dataset(f"hausdorff_distance_cat3", data=np.array(hd_cat3_list), compression='gzip')
            metrics.create_dataset(f"hausdorff_distance_cat4", data=np.array(hd_cat4_list), compression='gzip')
            metrics.create_dataset(f"hausdorff_distance_cat5", data=np.array(hd_cat5_list), compression='gzip')
            metrics.create_dataset(f"hausdorff_distance_cat6", data=np.array(hd_cat6_list), compression='gzip')
            metrics.create_dataset(f"hausdorff_distance_cat7", data=np.array(hd_cat7_list), compression='gzip')

        # metrics['dice'].attrs["id_list"] = id_list
        print(f"3d dice avg:{np.mean(dice_list)}, std: {np.std(dice_list)},{dice_list}")
        print(f"3d dice cat1 avg:{np.mean(dice_cat1_list)}, std: {np.std(dice_cat1_list)},{dice_cat1_list}")
        print(f"3d dice cat2 avg:{np.mean(dice_cat2_list)}, std: {np.std(dice_cat2_list)},{dice_cat2_list}")
        print(f"3d dice cat3 avg:{np.mean(dice_cat3_list)}, std: {np.std(dice_cat3_list)},{dice_cat3_list}")
        print(f"3d dice cat4 avg:{np.mean(dice_cat4_list)}, std: {np.std(dice_cat4_list)},{dice_cat4_list}")
        print(f"3d dice cat5 avg:{np.mean(dice_cat5_list)}, std: {np.std(dice_cat5_list)},{dice_cat5_list}")
        print(f"3d dice cat6 avg:{np.mean(dice_cat6_list)}, std: {np.std(dice_cat6_list)},{dice_cat6_list}")
        print(f"3d dice cat7 avg:{np.mean(dice_cat7_list)}, std: {np.std(dice_cat7_list)},{dice_cat7_list}")
        if len(metric_fns) > 8:
            print(f"3d surface distance avg:{np.mean(sd_list)}, std: {np.std(sd_list)},{sd_list}")
            print(f"3d surface distance cat1 :{np.mean(sd_cat1_list)}, std: {np.std(sd_cat1_list)},{sd_cat1_list}")
            print(f"3d surface distance cat2 :{np.mean(sd_cat2_list)}, std: {np.std(sd_cat2_list)},{sd_cat2_list}")
            print(f"3d surface distance cat3 :{np.mean(sd_cat3_list)}, std: {np.std(sd_cat3_list)},{sd_cat3_list}")
            print(f"3d surface distance cat4 :{np.mean(sd_cat4_list)}, std: {np.std(sd_cat4_list)},{sd_cat4_list}")
            print(f"3d surface distance cat5 :{np.mean(sd_cat5_list)}, std: {np.std(sd_cat5_list)},{sd_cat5_list}")
            print(f"3d surface distance cat6 :{np.mean(sd_cat6_list)}, std: {np.std(sd_cat6_list)},{sd_cat6_list}")
            print(f"3d surface distance cat7 :{np.mean(sd_cat7_list)}, std: {np.std(sd_cat7_list)},{sd_cat7_list}")
        if len(metric_fns) > 16:
            print(f"3d HD avg:{np.mean(hd_list)}, std: {np.std(hd_list)},{hd_list}")
            print(f"3d HD cat1 :{np.mean(hd_cat1_list)}, std: {np.std(hd_cat1_list)},{hd_cat1_list}")
            print(f"3d HD cat2 :{np.mean(hd_cat2_list)}, std: {np.std(hd_cat2_list)},{hd_cat2_list}")
            print(f"3d HD cat3 :{np.mean(hd_cat3_list)}, std: {np.std(hd_cat3_list)},{hd_cat3_list}")
            print(f"3d HD cat4 :{np.mean(hd_cat4_list)}, std: {np.std(hd_cat4_list)},{hd_cat4_list}")
            print(f"3d HD cat5 :{np.mean(hd_cat5_list)}, std: {np.std(hd_cat5_list)},{hd_cat5_list}")
            print(f"3d HD cat6 :{np.mean(hd_cat6_list)}, std: {np.std(hd_cat6_list)},{hd_cat6_list}")
            print(f"3d HD cat7 :{np.mean(hd_cat7_list)}, std: {np.std(hd_cat7_list)},{hd_cat7_list}")

        print(f"3D DICE: {np.mean(dice_list):.4f}({np.std(dice_list):.4f}), {np.mean(dice_cat1_list):.4f}({np.std(dice_cat1_list):.4f}), {np.mean(dice_cat2_list):.4f}({np.std(dice_cat2_list):.4f}), {np.mean(dice_cat3_list):.4f}({np.std(dice_cat3_list):.4f}), {np.mean(dice_cat4_list):.4f}({np.std(dice_cat4_list):.4f}), {np.mean(dice_cat5_list):.4f}({np.std(dice_cat5_list):.4f}), {np.mean(dice_cat6_list):.4f}({np.std(dice_cat6_list):.4f}), {np.mean(dice_cat7_list):.4f}({np.std(dice_cat7_list):.4f})")
        if len(metric_fns) > 8:
            print(f"3D SD: {np.mean(sd_list):.4f}({np.std(sd_list):.4f}), {np.mean(sd_cat1_list):.4f}({np.std(sd_cat1_list):.4f}), {np.mean(sd_cat2_list):.4f}({np.std(sd_cat2_list):.4f}), {np.mean(sd_cat3_list):.4f}({np.std(sd_cat3_list):.4f}), {np.mean(sd_cat4_list):.4f}({np.std(sd_cat4_list):.4f}), {np.mean(sd_cat5_list):.4f}({np.std(sd_cat5_list):.4f}), {np.mean(sd_cat6_list):.4f}({np.std(sd_cat6_list):.4f}), {np.mean(sd_cat7_list):.4f}({np.std(sd_cat7_list):.4f})")
        if len(metric_fns) > 16:
            print(f"3D HD: {np.mean(hd_list):.4f}({np.std(hd_list):.4f}), {np.mean(hd_cat1_list):.4f}({np.std(hd_cat1_list):.4f}), {np.mean(hd_cat2_list):.4f}({np.std(hd_cat2_list):.4f}), {np.mean(hd_cat3_list):.4f}({np.std(hd_cat3_list):.4f}), {np.mean(hd_cat4_list):.4f}({np.std(hd_cat4_list):.4f}), {np.mean(hd_cat5_list):.4f}({np.std(hd_cat5_list):.4f}), {np.mean(hd_cat6_list):.4f}({np.std(hd_cat6_list):.4f}), {np.mean(hd_cat7_list):.4f}({np.std(hd_cat7_list):.4f})")


        print(f"save at:{save_dir}/metrics.h5")
        # print(f"dice std:{np.mean(dice_list)},{np.std(dice_list)}, sensitivity std:{np.mean(sensitivity_list)},{np.std(sensitivity_list)}, specificity std:{np.mean(specificity_list)},{np.std(specificity_list)}, hausdorff std:{np.mean(hausdorff95_list)},{np.std(hausdorff95_list)}")
        # metrics.create_dataset(f"id_list", data=np.array(id_list))

        metrics.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-v', '--visual',action="store_true",
                      help='just visualize images')

    config = ConfigParser.from_args(args)
    main(config)

#python test_general_brats.py -c config_files/config_BRATS_seg_exp1_histnorm.json -r saved/models/brats_seg_AsynDGANv2_exp1/1006_175729/checkpoint-epoch15.pth


# Loading checkpoint: saved/models/brats_seg_AsynDGANv2_exp1/1006_175729/checkpoint-epoch15.pth ...
# Save test results at saved/test_results/brats_seg_AsynDGANv2_exp1/20201007153347
# {'loss': 0.25375720443743055, 'dice': 0.8487795651614011, 'sensitivity': 0.8259200434981685, 'specificity': 0.9962976476648352, 'hausdorff95': 13.69883527930403}
# save at:saved/test_results/brats_seg_AsynDGANv2_exp1/20201007153347/metrics.h5
# dice std:0.035784174933255745, sensitivity std:0.05968105867278165, specificity std:0.001003148295093025, hausdordd std:4.101768033350936

#     val_loss       : 0.9949499411630804
#     val_dice       : 0.8462186641408895
#     val_sensitivity: 0.821838926536415
#     val_specificity: 0.9963073654860342
#     val_hausdorff95: 13.494535079905301



# Loading checkpoint: saved/models/brats_seg_AsynDGANv2_exp2_histnorm_channelnorm/1007_113626/checkpoint-epoch4.pth ...
# Save test results at saved/test_results/brats_seg_AsynDGANv2_exp2_histnorm_channelnorm/20201007174811
# {'loss': 0.2523130500491286, 'accuracy': 0.9899026835794414, 'dice': 0.8127309945913461, 'sensitivity': 0.8494150462168041, 'specificity': 0.9936233688186813, 'hausdorff95': 17.155218349358975}
# save at:saved/test_results/brats_seg_AsynDGANv2_exp2_histnorm_channelnorm/20201007174811/metrics.h5
# dice std:0.0019223689652722636, sensitivity std:0.03683849609405132, specificity std:0.0451072621000645, hausdordd std:0.0015853996985996665

# Loading checkpoint: saved/models/brats_seg_AsynDGANv2_exp2_histnorm_channelnorm/1007_213349/checkpoint-epoch18.pth ...
# Save test results at saved/test_results/brats_seg_AsynDGANv2_exp2_histnorm_channelnorm/20201008102326
# {'loss': 0.2967462754729903, 'dice': 0.826373805231227, 'sensitivity': 0.8971841553628663, 'specificity': 0.9926958133012821, 'hausdorff95': 14.336575663919414}
# save at:saved/test_results/brats_seg_AsynDGANv2_exp2_histnorm_channelnorm/20201008102326/metrics.h5
# dice std:0.030393611120611865, sensitivity std:0.03422614613577275, specificity std:0.00175842465489657, hausdordd std:3.871668036322198

# ===================

# Loading checkpoint: saved/models/brats_seg_AsynDGANv2_exp13_nonorm/1007_231209/checkpoint-epoch28.pth ...
# Save test results at saved/test_results/brats_seg_AsynDGANv2_exp13_nonorm/20201010234316
# dice avg:0.7734367891571308, std: 0.18988324725919747,[0.23447647381470027, 0.8557775411183939, 0.8889396305894806, 0.8534080295062663, 0.7827208864769485, 0.8935648693866711, 0.8864946155943972, 0.8617606676834669, 0.918644308191613, 0.060320024076724994, 0.9216686599161108, 0.9113817834858894, 0.4675535200792602, 0.8922408146449574, 0.7731981169659323, 0.8431142511521417, 0.7652110532360189, 0.9180340612887757, 0.6492531411809435, 0.904271345702799, 0.7699728469680318, 0.8709843982028862, 0.8662880570225094, 0.387247764881143, 0.8620160580781874, 0.6599415018607621, 0.4258448309802429, 0.9022852776512139, 0.8936191421439258, 0.8302271902910117, 0.7576854200645968, 0.880537111732895, 0.8006178679322111, 0.9026041306208127, 0.8675335397094772, 0.7908500303565441, 0.8605595942451819, 0.627073568272296, 0.7394166539431871, 0.8263017799381066, 0.8738381691921684, 0.8068664164206137]
# sensitivity avg:0.7424329112536734, std: 0.2235034643782616,[0.17272199892263707, 0.7824917248370189, 0.8476763512581743, 0.8575311716324991, 0.9082955441640379, 0.9054815720590764, 0.8705278703644249, 0.9383765065190642, 0.926508494770246, 0.03137388747407388, 0.9682928830806474, 0.9077405023023087, 0.3143452476358391, 0.8950769715756843, 0.6672716484590415, 0.8825409229824586, 0.6463414634146342, 0.9307845345345346, 0.5467658049119071, 0.9528936814996065, 0.7062132546828915, 0.8120402815923226, 0.8139155863144556, 0.3552902985831671, 0.7818217011681864, 0.553066099237517, 0.2957734354824826, 0.9265553457489479, 0.9244222565977593, 0.7472186415023153, 0.6500212425242655, 0.8456038923188561, 0.7472954066250221, 0.8818307232455356, 0.8576977031469191, 0.8323494142606771, 0.8297250450327974, 0.4849197023110067, 0.6219886822958771, 0.7895631964538571, 0.8371589414858646, 0.934672639645648]
# specificity avg:0.9962062762096031, std: 0.0031310799047600335,[0.9877466906029241, 0.9981369774549881, 0.9974936492631861, 0.9957478179075931, 0.9909995763367971, 0.996795294443375, 0.9965294526099778, 0.9877138618709753, 0.9963055861107414, 0.999875479982302, 0.9937274900953998, 0.9973255626517195, 0.999714599891699, 0.9954382450812522, 0.9982368300556691, 0.9920191974223259, 0.9993752113469166, 0.9945190584633764, 0.9962841581154434, 0.9937174799705657, 0.9983372604427367, 0.9989835259845892, 0.9966032422322155, 0.9949129900752203, 0.9987004791863928, 0.9976482958218973, 0.9988837681078857, 0.9971273247145644, 0.9939490717134117, 0.9975622088680535, 0.9992944232093754, 0.9988073473753419, 0.9964261853963716, 0.9978619762267656, 0.9967434776933779, 0.9965536885241122, 0.9974409992229951, 0.9996667252101747, 0.9986448723930574, 0.9961150094971902, 0.9956461851338505, 0.9870523240965258]
# hausdorff95 avg:19.601467194702032, std: 13.955402815052762,[59.16528859693307, 16.32008206614018, 8.361477678411225, 14.229419645304052, 26.563826982277515, 11.907220716471366, 11.098442309568007, 16.79906110667347, 9.48750461461126, 73.99228029486379, 11.38112119452824, 12.42607411340532, 24.5923896065146, 18.393876643502363, 17.67200668201762, 13.614782037360316, 11.405523212972303, 9.633174319970863, 38.273989741738326, 11.299396759473078, 17.275987226755035, 9.663830958720116, 12.336661604586302, 53.91861429259118, 13.112102597315207, 19.1238461056772, 39.44778825105114, 9.517148977935907, 19.725918944325304, 19.543439353550635, 14.187287874757278, 7.981897425079276, 17.122154230786425, 10.009427587492231, 20.930488019163086, 22.2296676209678, 14.624243526498526, 6.348590070177108, 17.650940115136343, 16.546370586670484, 17.446819222930223, 27.90145926258151]

# Loading checkpoint: saved/models/brats_seg_AsynDGANv2_exp2_histnorm_channelnorm/1007_213349/checkpoint-epoch18.pth ...
# Save test results at saved/test_results/brats_seg_AsynDGANv2_exp2_histnorm_channelnorm/20201010235615
# dice avg:0.7961252998890717, std: 0.10646356099328659,[0.9077686292510684, 0.8983361076064715, 0.5433564586620975, 0.9064842098913736, 0.6974094718391626, 0.576387396783177, 0.8591826079682268, 0.8258005376775309, 0.8560599391468942, 0.8327857963684514, 0.6391832571029595, 0.607585050180458, 0.8610419283249702, 0.8924926294437583, 0.9159821552317949, 0.8168500973117859, 0.8451433080108035, 0.8282751688706708, 0.895104573772331, 0.6472106101001118, 0.9055112802933917, 0.8274599825354706, 0.8816590418532119, 0.6601729001629957, 0.8261089916246108, 0.8375763323803013, 0.8382868906249452, 0.8318167359011854, 0.842168996956479, 0.8398749684226222, 0.6157279517520113, 0.7910962512835683, 0.8189683608727423, 0.8695181402994583, 0.6014145777952585, 0.8410885664676655, 0.8167074170424243, 0.8971094297375776, 0.7454503709455842, 0.5974262211235638, 0.8443825775596335, 0.8552966761622098]
# sensitivity avg:0.8703536047990712, std: 0.12406168004861604,[0.9177078420745142, 0.9775304514927987, 0.4364562586180481, 0.9332971767754377, 0.9281160384023939, 0.5563982349696636, 0.9301029192042533, 0.9547230623192565, 0.9860803180634099, 0.783995377452768, 0.8223295039709148, 0.6681457095827668, 0.9048513633296731, 0.9258979143205023, 0.9545789978599365, 0.9282044887780548, 0.8932607991201627, 0.9331539381513754, 0.9072215809658563, 0.5367429916541836, 0.9611176380446442, 0.8750153793736573, 0.9612949563736966, 0.883127264975485, 0.9552697945228665, 0.9082677934205621, 0.8004902690011577, 0.9218023691707902, 0.9744602802333273, 0.95304312777106, 0.7852743561030235, 0.8518644838562983, 0.9297070084900949, 0.9674178304634281, 0.8440397720925036, 0.9171605521821862, 0.89325029194239, 0.9149086669257676, 0.7942563672571009, 0.6809925580764098, 0.9699623408452388, 0.9333333333333333]
# specificity avg:0.9929957061949077, std: 0.004403142569201737,[0.9970981096542845, 0.9949887960065793, 0.9986611294518953, 0.9964378828751467, 0.9813414460606606, 0.9985665180014494, 0.9941625623606136, 0.9959024865164102, 0.9852456583352696, 0.9956842965850163, 0.9890133828527742, 0.994551425805064, 0.992474917937995, 0.9927388785592327, 0.9950883079838168, 0.9959770576947946, 0.9941346983009005, 0.9900068381283798, 0.9941978730115812, 0.9977299302479621, 0.9938082941787872, 0.9937996543107334, 0.9955736916463763, 0.994810494202079, 0.9821958819730512, 0.9946862843857146, 0.9964491803643707, 0.9945483700336856, 0.9864176888864524, 0.9955556369046114, 0.9941339819267286, 0.9951586230377315, 0.9960508874529455, 0.9910211549031395, 0.9816425958405988, 0.9938216413479839, 0.9900652257927129, 0.995104936475786, 0.9967002191750239, 0.9848488693221656, 0.9915657469411047, 0.9938584047145173]
# hausdorff95 avg:14.916177909653626, std: 7.455832668502178,[7.668612662552544, 10.415200616639988, 19.249126756847204, 5.318664236714689, 24.309462683622936, 6.597414033905644, 9.680199704686268, 6.110722014373502, 22.63386465723176, 14.722926790174535, 9.399827902023663, 44.94958764397731, 14.242332152401715, 7.843238232516493, 10.666393487889206, 29.49670390882404, 12.367046563852607, 13.118381706419152, 6.7877843962754945, 25.80417862718714, 11.654740342522294, 8.977508302110861, 18.279274994851697, 14.938562631981824, 20.81930013899203, 9.071949293570292, 11.117190980546164, 19.41103971335648, 16.60349591509234, 12.159983063332668, 16.276506302988007, 7.606723222831648, 11.03813506346552, 12.95664695983144, 15.987636902446836, 14.635761329053244, 22.643176909101676, 15.583553395435374, 21.067820467376464, 14.482602232714619, 21.07019853995523, 8.715996725779704]

# Loading checkpoint: saved/models/brats_seg_AsynDGANv2_exp1/1006_175729/checkpoint-epoch15.pth ...
# Save test results at saved/test_results/brats_seg_AsynDGANv2_exp1/20201007153347
# 3d dice avg:0.8085003917675293, std: 0.1629325519683161,[0.17467258126507795, 0.8586126108193958, 0.8856865151073057, 0.8614948721420593, 0.8767115148756597, 0.8910858472031082, 0.9268276149640852, 0.8661561244191885, 0.8881280394446315, 0.4104437776377442, 0.9017422683386084, 0.9165098812408515, 0.3967212267390485, 0.9087254392099517, 0.7413319262870464, 0.8729335458566538, 0.8659281527180814, 0.9312520573299587, 0.7764307892245857, 0.9017139688225582, 0.8619697844626714, 0.8833985627983351, 0.8907576055435839, 0.7321628369943588, 0.8493585610540219, 0.620302653564166, 0.49418894305725547, 0.9038057950415782, 0.9211996193879566, 0.8938684237226286, 0.756813451937319, 0.8825757106108516, 0.8630514269725283, 0.9150238911340892, 0.8652817182482359, 0.8477022220627426, 0.892580160875679, 0.6083228872365437, 0.8262793423278757, 0.8774786390143968, 0.8824300574268765, 0.8353554071169403]
# 3d sensitivity avg:0.7858910344250626, std: 0.21481501669541048,[0.10279285625492064, 0.7826214371159675, 0.8763236777738262, 0.9038195114252144, 0.9097372831230284, 0.9303674124457861, 0.957901127189731, 0.9458514863787029, 0.9063685000400967, 0.26249048281656123, 0.982325667275301, 0.9219209989971208, 0.24901426510658758, 0.923383071507382, 0.6131461380056542, 0.9024895686515302, 0.8407812863940372, 0.955686936936937, 0.6786912990976771, 0.9757507937612773, 0.8809695880618005, 0.8318969599393882, 0.8958144068622673, 0.6084604199776823, 0.7585312135432197, 0.4923386634122936, 0.35565693954262434, 0.9274809631357449, 0.9671494867107299, 0.8626994785373895, 0.7192065100166672, 0.8566432318618179, 0.8418494360914602, 0.8909958357816212, 0.8938883189212956, 0.8371460197398763, 0.8620127111443429, 0.45162553858206034, 0.8015844785772029, 0.8377454903650804, 0.8366822813938198, 0.9755816733569065]
# 3d specificity avg:0.996489072691594, std: 0.0028271174665355737,[0.9969753612654996, 0.9983733397292461, 0.9956799393778709, 0.9945652946455836, 0.9963873034374785, 0.9958272641979874, 0.9959462948562667, 0.9877788515703136, 0.9945077734728051, 0.9997675380734691, 0.9907289940645031, 0.9971472554204885, 0.9999402018820702, 0.9955367815720608, 0.9987683219316232, 0.9937500692823823, 0.9985295817955502, 0.9945305128866504, 0.9981210130293385, 0.9923436477334052, 0.9978839472656651, 0.9990048740737216, 0.9939780961452134, 0.9994312303319915, 0.9988833731402871, 0.9981827921682429, 0.9989990628867822, 0.9971799857842975, 0.9944475912673505, 0.9968816009688971, 0.9980543185470656, 0.9986557373888185, 0.9967397020295488, 0.9983259245462165, 0.9953114286349395, 0.9982559990484087, 0.9981966166331964, 0.9998206664226178, 0.9968888301928962, 0.9977078802074619, 0.9967090701944079, 0.9877969849443349]
# 3d hausdorff95 avg:14.259702019192828, std: 9.109190045419563,[57.83429485147951, 13.908888441157426, 7.925123000445429, 8.212030559750936, 21.350477504410385, 8.511707038083468, 5.65409983953806, 12.438350865246885, 13.77604142395169, 24.893762838038587, 7.424979607650032, 7.754792321106877, 17.749764059207184, 7.9488176632295335, 15.172792390962377, 8.805437995860999, 9.766742482424124, 10.668596373572615, 18.280658170063102, 10.557923339553883, 6.403227446027278, 9.904884525642574, 16.893777918561003, 19.469678890805543, 14.021198180233693, 25.281815791231928, 36.27705374165476, 8.792059820519787, 8.67838955941138, 15.17684658249427, 17.64251327891753, 9.571986208347997, 14.510196554731829, 8.599010587244708, 17.70765578987306, 17.41826391151142, 9.795403098334834, 6.6667434544608115, 10.690159227472975, 12.12450682840744, 8.571036140157657, 16.075796504323197]
# save at:saved/test_results/brats_seg_AsynDGANv2_exp1/20201011232032/metrics.h5

# Modality bank (pretrain LGG)
# ======================
# {'loss': 0.3318147253204178, 'dice': 0.8573864611950549, 'sensitivity': 0.834233881353022, 'specificity': 0.9975524231627747, 'hausdorff95': 13.964759329212454}
#
# 3d dice avg:0.8529040185868366, std: 0.10986532531868715,[0.8556415387759169, 0.9299951681998767, 0.9348406267307148, 0.8364977743363753, 0.8447542129126547, 0.9300607403413589, 0.8955015153626199, 0.5944733523360997, 0.9101016056330813, 0.9031562197879689, 0.8173975671906393, 0.9345034017816567, 0.9105652265397453, 0.874563521236159, 0.9410863964080809, 0.9066201018745031, 0.936612544959981, 0.8745101971980724, 0.8981114709581394, 0.7052644855399616, 0.8792555308896433, 0.9141910323757121, 0.9271298866339804, 0.8958587198051619, 0.5716277924700558, 0.857092084732627, 0.5850587961522481, 0.8552790526593324, 0.9218957220028108, 0.8620721572452084, 0.9526356881206448, 0.9185184350248088, 0.5405911798568582, 0.9313511267696891, 0.9418425869522739, 0.865394114083723, 0.921710476283735, 0.9085365848280034, 0.7833737059101142, 0.8689650722998954, 0.621941491072677, 0.8633898763743361]
# 3d sensitivity avg:0.8245268946775136, std: 0.17167373400422495,[0.9666410801696259, 0.9103828828828829, 0.9364722638943208, 0.7559070557861368, 0.7991738651111514, 0.9801099645077241, 0.9385272502669542, 0.42481166853662444, 0.9130263227118882, 0.8521220391851058, 0.7673553579278396, 0.9625554193540996, 0.8704922748191979, 0.8278302688373042, 0.942056907139174, 0.9511675766984286, 0.9544013457992301, 0.8243204848944029, 0.8770505440388646, 0.5457305131218174, 0.8898464025869038, 0.969831292153079, 0.9357744949214459, 0.904642850167002, 0.43046548872377854, 0.7646538844023386, 0.4147083297466357, 0.7592950076117834, 0.8880787698451796, 0.7719451328051377, 0.942882507728074, 0.9818596467320511, 0.37618257867342575, 0.8845951767981927, 0.9610053060197357, 0.7954734347103367, 0.9298151387366725, 0.945571273659306, 0.9620191864219167, 0.8972549989065635, 0.45390544068288236, 0.7701881487403601]
# 3d specificity avg:0.9976082156602706, std: 0.0021330061320949875,[0.99007850810301, 0.9973192407159778, 0.9975110653641799, 0.9994486476444846, 0.9964854697028802, 0.9939807340879973, 0.9957106924619787, 0.9999586245345637, 0.9973893807695494, 0.9981828412827084, 0.9988312608146029, 0.9960255682680698, 0.9977102729984108, 0.9983055384308734, 0.9974875792086728, 0.9976623328013121, 0.9973472949150268, 0.9988234153160951, 0.998894344199286, 0.9999899488555449, 0.9969875168619962, 0.9959833818106308, 0.998128999559486, 0.9985085515034972, 0.9989386326302128, 0.9992084477154454, 0.9999646566600406, 0.9993423014875586, 0.9988469031098061, 0.9994307876557672, 0.998914046712081, 0.9919920761926072, 0.9997025480981918, 0.9993850892212925, 0.9967627065887176, 0.9986273361611668, 0.9963019400365678, 0.9970340887585163, 0.9937537244420188, 0.9954646382307825, 0.9997661788522287, 0.9993577449675338]
# 3d hausdorff95 avg:14.663442412056018, std: 9.929328359621941,[11.5117820118888, 8.840594799579154, 8.144562710686392, 11.078596345876019, 22.3884431270881, 6.107548061006047, 10.688895152825, 24.888000340467503, 4.822941514017487, 7.0571030480707275, 17.568987043482952, 10.777331122336028, 7.387172623818394, 23.677913949053472, 11.612582429617852, 6.03420314565835, 7.749109708474739, 7.840172583501187, 6.863449717710675, 6.7667825752563155, 22.33368024600176, 7.419611935897013, 24.42892533390343, 5.704510535586121, 59.43812701284338, 16.004958067312625, 20.34531924771873, 11.240570759804276, 11.885519046261347, 11.413573527165555, 9.070081226998193, 8.676710926522517, 23.019163361904063, 8.440477220539282, 12.548621497703383, 14.868720630717768, 10.030458685583202, 22.540080014046765, 30.428155081113847, 31.69657397245603, 18.65270885168114, 13.871862114177082]
# save at:saved/test_results/modalitybank_rebuttal_lgg_t1t2flair_epoch35/20210519101830/metrics.h5

# Modality bank (pretrain M&Ms)
# ======================
# {'loss': 0.3450307570970975, 'dice': 0.8440794557005494, 'sensitivity': 0.8181288275526557, 'specificity': 0.9974195319654304, 'hausdorff95': 17.182765281593408}
# 3d dice avg:0.8406419578204359, std: 0.12019776710449294,[0.8990815210934368, 0.931377443814748, 0.9336582193473748, 0.8529674476489666, 0.8299575685514431, 0.9265314817862653, 0.8895935363087647, 0.6771394483753354, 0.9004812576422959, 0.8924573417468513, 0.7936530870827103, 0.9293703738011632, 0.9078883487981474, 0.8380958355543966, 0.9405901123529695, 0.8826503777108817, 0.9335547011842957, 0.7180662962380572, 0.9007985405932516, 0.7803267445760417, 0.86318701695343, 0.9213454471666019, 0.9022983872871477, 0.9002210104597216, 0.5399325466083715, 0.7607947473844077, 0.6520656077027945, 0.7655721108292106, 0.9091611956825698, 0.8761189004202571, 0.9318985085483532, 0.9006037240142925, 0.3572214623359944, 0.9258602954310303, 0.9420281144593946, 0.8669437796418112, 0.9058420641357573, 0.9095522793510409, 0.834630223487778, 0.8379663696661034, 0.606210021509095, 0.8392687311757596]
# 3d sensitivity avg:0.807490153608468, std: 0.1700720147329049,[0.9413859492380676, 0.9208033033033033, 0.9226436130779693, 0.7801562142553679, 0.7833977588300144, 0.9812332027162874, 0.8899101777600972, 0.5278410001602821, 0.8736474755800291, 0.8360113071449459, 0.7425016061948398, 0.9623549359025765, 0.8683473044049967, 0.7940981545049791, 0.9456804012562696, 0.9084853026896494, 0.9383391025848403, 0.7884742873378161, 0.8877150367683758, 0.6711711711711712, 0.862586903799515, 0.9658380939643478, 0.8989966307527121, 0.922886106607809, 0.40770301136811155, 0.6154271137096271, 0.4882212848985682, 0.6211895307897944, 0.8496397159766423, 0.8066472374437192, 0.9228278363947576, 0.992508988011976, 0.22346695038124145, 0.8750262701623496, 0.9653582768018247, 0.7977051653019762, 0.9032944062592994, 0.9363786474763407, 0.892929619038834, 0.7849067726507754, 0.4875440268511996, 0.7293065580323658]
# 3d specificity avg:0.9974878424252618, std: 0.002155138767897065,[0.9948253212828888, 0.9968078219286873, 0.9980033892521641, 0.9994731833500702, 0.9960507722412283, 0.993540341332228, 0.9969837199533309, 0.9997061435921938, 0.9981341866127093, 0.998045873604845, 0.9986361721012381, 0.995571093937929, 0.9975415538848936, 0.99738146706637, 0.9972724699838237, 0.9976154509102176, 0.9977177701123541, 0.9921240799736015, 0.9987898499297808, 0.9997349671909485, 0.9969476068119925, 0.9965426769471883, 0.9978849196807235, 0.9983463824518897, 0.9985617569553455, 0.9999022416474199, 0.9998895091702241, 0.9999343783267278, 0.9994191298970522, 0.9989564003776238, 0.9982897702708126, 0.9891428126019963, 0.9994710858609269, 0.9993783585866745, 0.9965804280240025, 0.9986391461386295, 0.9961624944611667, 0.9973252227995321, 0.99688021494768, 0.997610033389655, 0.9950684687113853, 0.9996007155608361]
# 3d hausdorff95 avg:18.329730030288623, std: 12.4069128733122,[10.057664827792092, 7.554752032248484, 5.4502763538295715, 8.432324004301872, 32.581259126732704, 8.294346582831027, 15.275316322398224, 21.233546403185176, 6.683900667253341, 7.386196692421614, 37.93352929535497, 12.359571976712038, 7.660862703037031, 36.53918751684346, 14.771335004766081, 14.688967772394987, 9.861146345135664, 21.342625016231523, 11.758435618254632, 10.928272415241882, 29.466716764884673, 7.5126287077446126, 33.91478550610321, 11.575357257456062, 70.06207393006396, 17.44949037325598, 31.037403633657277, 15.149156047090973, 13.257342369035912, 12.599606646388462, 10.611058677626, 15.962324467044793, 34.17333851187698, 8.130646812535021, 10.74165046210257, 15.127298964079394, 11.349065798816873, 28.619950887164354, 31.546022378440238, 28.425151446273915, 17.663557157192944, 14.680517796321658]
# save at:saved/test_results/modalitybank_rebuttal_mms_t1t2flair_epoch35/20210519104053/metrics.h5

# 3d surface distance avg:1.5312735804849087, std: 0.7379668263185027,[1.1741046923177596, 0.3859747909701627, 1.161515326722822, 1.7722211032585922, 1.9576957564874016, 2.7361298131527136]
# 3d surface distance cat1 :1.5312735804849087, std: 0.7379668263185027,[1.1741046923177596, 0.3859747909701627, 1.161515326722822, 1.7722211032585922, 1.9576957564874016, 2.7361298131527136]
# 3d surface distance cat2 :0.3181172565001378, std: 0.28061634963602716,[0.2406333556086852, 0.1496022274719891, 0.9305182439093431, 0.16096187908892634, 0.304178191838858, 0.12280964108302495]
# 3d surface distance cat3 :1.6352671202184021, std: 1.695499154961995,[0.4884608171867364, 0.5335328640447593, 1.210445643460044, 5.122211442589042, 2.2406788164092237, 0.21627313762060793]
# 3d surface distance cat4 :1.217371630868681, std: 0.9883094926672057,[2.22064857928889, 0.17784350037353233, 1.9475275159877279, 0.34562471352773916, 2.4163527182420723, 0.19623275779212523]
# 3d surface distance cat5 :1.8098993616165346, std: 1.2381130092091726,[3.32570694746496, 0.6942171756944729, 1.4776263724546537, 3.513090563409709, 1.6577124775102425, 0.19104263316516795]
# 3d surface distance cat6 :1.3709321573133906, std: 1.382269708036026,[0.6275304897048217, 0.3344488810734946, 0.8528717181099547, 2.113781024022698, 4.124296703526976, 0.17266412744239923]
# 3d surface distance cat7 :0.35994775473792545, std: 0.29505372098448773,[0.10044868672478688, 0.31434382899233304, 0.5488385754272674, 0.11674244804108594, 0.9227372168785711, 0.15657577236350806]

# exp1-6
# 3D DICE: 0.864155406558504(0.06242003188799522), 0.9021916936672039(0.09103419342905492), 0.8430905322436693(0.0909738744580681), 0.8979921587332057(0.054650939828004105), 0.8382819997684602(0.08992607194778646), 0.8705526726463152(0.06172765331619442), 0.928983013558016(0.03404477708229265), 0.767995775292658(0.1306145048879155)
# 3D SD: 4.444237365012086(1.901827622412228), 2.461474025361923(3.6554996142715366), 4.213811414760902(1.5674134653545273), 6.085316058825729(5.467286961600807), 6.316942506447858(4.515850614151271), 2.2801670479864806(1.0209647650517268), 2.3857015415100604(1.8162837741753421), 7.366248960191645(7.324189968329248)


