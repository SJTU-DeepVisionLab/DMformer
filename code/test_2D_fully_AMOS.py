import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from config import get_config
# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/AbdomenMR', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='AbdomenMR/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='swinunet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=14,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=15,
                    help='labeled data')
# parser.add_argument('--eval', type=str, default=True,
#                     help='evaluation')
parser.add_argument('--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
#args = parser.parse_args()
args = parser.parse_args()
config = get_config(args)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    if np.sum(gt == 1)==0:
        asd = 0
    else:
        asd = metric.binary.asd(pred, gt)
    #hd95 = metric.binary.hd95(pred, gt)
    return dice, asd, dice


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224 ), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    if np.sum(prediction == 4)==0:
        forth_metric = 0,0,0
    else:
        forth_metric = calculate_metric_percase(prediction == 4, label == 4)

    if np.sum(prediction == 5)==0:
        fifth_metric = 0,0,0
    else:
        fifth_metric = calculate_metric_percase(prediction == 5, label == 5)

    if np.sum(prediction == 6)==0:
        sixth_metric = 0,0,0
    else:
        sixth_metric = calculate_metric_percase(prediction == 6, label == 6)

    if np.sum(prediction == 7)==0:
        seventh_metric = 0,0,0
    else:
        seventh_metric = calculate_metric_percase(prediction == 7, label == 7)

    if np.sum(prediction == 8)==0:
        eighth_metric = 0,0,0
    else:
        eighth_metric = calculate_metric_percase(prediction == 8, label == 8)

    if np.sum(prediction == 9)==0:
        nineth_metric = 0,0,0
    else:
        nineth_metric = calculate_metric_percase(prediction == 9, label == 9)

    if np.sum(prediction == 10)==0:
        tenth_metric = 0,0,0
    else:
        tenth_metric = calculate_metric_percase(prediction == 10, label == 10)

    if np.sum(prediction == 11)==0:
        eleventh_metric = 0,0,0
    else:
        eleventh_metric = calculate_metric_percase(prediction == 11, label == 11)

    if np.sum(prediction == 12)==0:
        twelveth_metric = 0,0,0
    else:
        twelveth_metric = calculate_metric_percase(prediction == 12, label == 12)

    if np.sum(prediction == 13)==0:
        thirteth_metric = 0,0,0
    else:
        thirteth_metric = calculate_metric_percase(prediction == 13, label == 13)

    #nineth_metric = calculate_metric_percase(prediction == 2, label == 2)
    #third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric, forth_metric, fifth_metric, sixth_metric, seventh_metric, eighth_metric, nineth_metric, tenth_metric, eleventh_metric, twelveth_metric, thirteth_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}/{}".format(
        FLAGS.exp, FLAGS.model)
    test_save_path = "../model/{}/{}_labeled_{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    #net = net_factory(net_type=FLAGS.model, in_chns=1,
                      #class_num=FLAGS.num_classes)
    net = ViT_seg(config, num_classes=FLAGS.num_classes).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    forth_total = 0.0
    fifth_total = 0.0
    sixth_total = 0.0
    seventh_total = 0.0
    eighth_total = 0.0
    nineth_total = 0.0
    tenth_total = 0.0
    eleventh_total = 0.0
    twelveth_total = 0.0
    thirteth_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric, forth_metric, fifth_metric, sixth_metric, seventh_metric, eighth_metric, nineth_metric, tenth_metric, eleventh_metric, twelveth_metric, thirteth_metric  = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        forth_total += np.asarray(forth_metric)
        fifth_total += np.asarray(fifth_metric)
        sixth_total += np.asarray(sixth_metric)
        seventh_total += np.asarray(seventh_metric)
        eighth_total += np.asarray(eighth_metric)
        nineth_total += np.asarray(nineth_metric)
        tenth_total += np.asarray(tenth_metric)
        eleventh_total += np.asarray(eleventh_metric)
        twelveth_total += np.asarray(twelveth_metric)
        thirteth_total += np.asarray(thirteth_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list), forth_total / len(image_list), fifth_total / len(image_list), sixth_total / len(image_list), seventh_total / len(image_list), eighth_total / len(image_list), nineth_total / len(image_list), tenth_total / len(image_list), 
                  eleventh_total / len(image_list), twelveth_total / len(image_list), thirteth_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    #config = get_config(FLAGS)
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2]+metric[3]+metric[4]+metric[5]+metric[6]+metric[7]+metric[8]+metric[9]+metric[10]+metric[11]+metric[12])/13)
