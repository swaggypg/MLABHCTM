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
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler,WeakStrongAugment,CTATransform)
#from val_2D import test_single_volume,test_single_volume_double_net
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/linruohan/datasets/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Teaching_Between_CNN_Transformer', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
#parser.add_argument("--load", default=True, action="store_true", help="restore previous checkpoint") ###
parser.add_argument(
    '--cfg', type=str, default="/home/linruohan/unet-vit/code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
# parser.add_argument(
#     '--cfg', type=str, default="./code/configs/swin_tiny_patch4_window7_224_lite_otherCKPT.yaml", help='path to config file', ) ###
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
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

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

args = parser.parse_args()
config = get_config(args)



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=[10, 1, 1])
    return dice, hd95



# def calculate_metric_percase(pred, gt):  # if raise RuntimeError('The first supplied array does not contain any binary object.')
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if np.count_nonzero(pred) == 0:
#         pred = np.ones_like(pred)
#         gt = gt.astype(int)
#         gt[gt == 0] = 2
#         gt -= 1
#         gt = gt.astype(bool)
#     dice = metric.binary.dc(pred, gt)
#     asd = metric.binary.asd(pred, gt)
#     hd95 = metric.binary.hd95(pred, gt)
#     return dice, hd95, asd


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
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "/home/linruohan/model/ACDC/Cross_Teaching_Between_CNN_Transformer_7/unet_30000_umcO_8_fixmatch"
    test_save_path = "/home/linruohan/unet-vit/model/test/umc8_predictions/test1"
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net1 = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        #snapshot_path, 'unet_best_model1.pth'.format(FLAGS.model))
        snapshot_path, 'model1_iter_27200_dice_0.8918.pth'.format(FLAGS.model))
        #snapshot_path, 'model1_iter_24000.pth'.format(FLAGS.model))
    net1.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net1.eval()


    # net2 = ViT_seg(config, img_size=args.patch_size,
    #                  num_classes=args.num_classes).cuda()
    # net2.load_state_dict(torch.load("/home/linruohan/unet-vit/model/ACDC/Cross_Teaching_Between_CNN_Transformer_7/unet_30000_umcO_8_fixmatch_No4S 0.8890 0.8858/model2_iter_30000.pth"))
    # #net2.load_from(config)
    # net2.eval()

    # db_test = BaseDataSets(base_dir=args.root_path,split="test")
    # testloader = DataLoader(db_test, batch_size=1,shuffle=False,num_workers=1)
    # metric_list_test = 0.0
    # for i_batch, sampled_batch in enumerate(testloader):
    #     metric_i = test_single_volume_double_net(
    #         sampled_batch["image"], sampled_batch["label"],net1,net2, classes=4, patch_size=args.patch_size)
    #     metric_list_test += np.array(metric_i)
    # metric_list_test = metric_list_test / len(db_test)
    # performance2_test = np.mean(metric_list_test , axis=0)
    # print(performance2_test)
    
    metric_list = []  # 存储每个图像的性能指标
    mean_list = []
    for case in image_list:
        first_metric, second_metric, third_metric = test_single_volume(
            case, net1, test_save_path, FLAGS)
        metric_list.append([first_metric, second_metric, third_metric])
        mean_list.append((np.array(first_metric)+ np.array(second_metric) + np.array(third_metric)) / 3)
        #print(case)
        #print([first_metric[1]+second_metric[1]+third_metric[1]])
    
    metric_array = np.array(metric_list)  # 转换为 numpy 数组
    mean_array = np.array(mean_list)

    std_mean = np.std(mean_array, axis=0)

    avg_metric = np.mean(metric_array, axis=0)  # 计算每个性能指标的平均值


    avg_metric_overall = np.mean(avg_metric, axis=0)  # 计算平均值的平均值
    
    #np.set_printoptions(suppress=True)
    return avg_metric#,std_mean



    # first_total = 0.0
    # second_total = 0.0
    # third_total = 0.0
    # for case in tqdm(image_list):
    #     first_metric, second_metric, third_metric = test_single_volume(
    #         case, net1, test_save_path, FLAGS)
    #     first_total += np.asarray(first_metric)
    #     second_total += np.asarray(second_metric)
    #     third_total += np.asarray(third_metric)
    #     print(case)
    #     print([first_metric[1],second_metric[1],third_metric[1]])
    # avg_metric = [first_total / len(image_list), second_total /
    #               len(image_list), third_total / len(image_list)]
    # avg_metric = np.array(avg_metric)
    # return avg_metric





# if __name__ == '__main__':
#     # FLAGS = parser.parse_args()
#     # Inference(FLAGS)

#     FLAGS = parser.parse_args()
#     metric,metric_std= Inference(FLAGS)
#     print(metric)
#     print(metric_std)
#     #print(metic3)
#     #print((metric[0]+metric[1]+metric[2])/3)


#     # FLAGS = parser.parse_args()
#     # metric = Inference(FLAGS)
#     # print(metric)
#     # print((metric[0]+metric[1]+metric[2])/3)

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)