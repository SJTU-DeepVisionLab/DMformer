

import argparse
import logging
import os
import random
import shutil
import sys
import time
import ipdb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.mask import BlockMaskGenerator
from config import get_config
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, metrics, ramps
from val_2D import test_single_volume
from pcgrad import PCGrad

# wrap your favorite optimizer



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/DMformer', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='swinunet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.05,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
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
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
# masks
parser.add_argument('--mask_thershold', type=float,
                    default=0.25, help='mask_ratio')

parser.add_argument('--mask_class_ratio', type=float,
                    default=0.25, help='mask_ratio')

parser.add_argument('--adjustment_rate_intra', type=float,
                    default=0.00012, help='adjustment_rate')

parser.add_argument('--adjustment_rate_inter', type=float,
                    default=0.00012, help='adjustment_rate')


parser.add_argument('--mask_ratio', type=float,
                    default=0.6, help='mask_ratio')

parser.add_argument('--mask_ratio_stronger', type=float,
                    default=0.4, help='mask_ratio')

parser.add_argument('--mask_size', type=float,
                    default=16, help='mask_size')
args = parser.parse_args()
config = get_config(args)




def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1":13, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate" in dataset:
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "11": 306, "14": 391, "18": 478, "35": 940}

    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):

    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def update_ema_average_confidence(current1, average1, current2, average2, alpha, global_step):

    alpha = min(1 - 1 / (global_step + 1), alpha)
    average1 = average1 * alpha + current1 * (1 - alpha)
    average2 = average2 * alpha + current2 * (1 - alpha)

    return average1, average2


def update_confidence(args, outputs_soft1_intra, input_mask_intra, outputs_soft1_inter, input_mask_inter, outputs_soft2, current_average_confidence_intra, increase_intra, average_confidence_intra, current_average_confidence_inter, average_confidence_inter):

    all_existing_class = torch.unique(torch.argmax(outputs_soft2[args.labeled_bs:], dim=1)* (1-input_mask_intra[args.labeled_bs:].squeeze(1).long()))
    all_existing_class_inter = torch.unique(torch.argmax(outputs_soft1_intra[args.labeled_bs:], dim=1))

    prediction_class_intra = torch.argmax(outputs_soft1_intra[args.labeled_bs:], dim=1)
    prediction_class_inter = torch.argmax(outputs_soft1_inter[args.labeled_bs:], dim=1)
    for class_i in range(args.num_classes):

        if class_i in all_existing_class:

            current_average_confidence_intra[class_i] = (torch.sum((outputs_soft1_intra[args.labeled_bs:][:, class_i, :, :])*(prediction_class_intra==class_i) * (1-input_mask_intra[args.labeled_bs:].squeeze(1).long()))) / (torch.sum((prediction_class_intra==class_i) * (1-input_mask_intra[args.labeled_bs:].squeeze(1).long())) + 1e-5)

            increase_intra[class_i] = current_average_confidence_intra[class_i] > average_confidence_intra[class_i]
        else:
            current_average_confidence_intra[class_i] = average_confidence_intra[class_i]


        if class_i in all_existing_class_inter:

            current_average_confidence_inter[class_i] = (torch.sum((outputs_soft1_inter[args.labeled_bs:][:, class_i, :, :])*(prediction_class_inter==class_i))) / (torch.sum((prediction_class_inter==class_i))+ 1e-5)#(torch.sum(torch.max(outputs_soft1_inter[args.labeled_bs:], dim=1)[0]*(prediction_class_inter==class_i) * (input_mask_intra[args.labeled_bs:]-input_mask_inter[args.labeled_bs:]).squeeze(1).long()))/(torch.sum((input_mask_intra[args.labeled_bs:]-input_mask_inter[args.labeled_bs:]).squeeze(1).long()*(prediction_class_inter==class_i)) + 1e-5)
        else:
            current_average_confidence_inter[class_i] = average_confidence_inter[class_i]

    increase_inter = np.mean(current_average_confidence_inter) > (np.mean(average_confidence_inter + 1e-5))
    increase_inter_num = np.mean(current_average_confidence_inter) - (np.mean(average_confidence_inter + 1e-5))
    increase_intra_num = current_average_confidence_intra - average_confidence_intra
    return increase_intra, increase_intra_num, increase_inter, increase_inter_num


@torch.no_grad()
def adjust_parameters_by_confidence2(args, confidence_increase_intra,increase_intra_num, confidence_increase_inter, increase_inter_num, mask_threshold, mask_class_ratio, alpfa):
    adjustment_rate_intra = args.adjustment_rate_intra*np.round(np.tanh(increase_intra_num/(args.adjustment_rate_intra)),decimals=4)#*(1.0 - alpfa)**2#**0.5
    adjustment_rate_inter = args.adjustment_rate_inter*np.round(np.tanh(increase_inter_num/(args.adjustment_rate_inter)),decimals=4)#*(1.0 - alpfa)**2#*10#**0.5

    mask_threshold  = np.round(np.clip(mask_threshold + adjustment_rate_intra, args.mask_thershold, 0.75),decimals=4)

    mask_class_ratio = np.round(np.clip(mask_class_ratio + adjustment_rate_inter, args.mask_class_ratio, 0.75),decimals=4)


    return mask_threshold, mask_class_ratio


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    mask_gen = BlockMaskGenerator(args.mask_ratio, args.mask_ratio_stronger, args.mask_size)
    def create_model(ema=False):

        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    def create_model_swinunet(ema=False):

        model = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    model1 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()

    model2 = create_model_swinunet(ema=True)

    model1.load_from(config)

    model2.load_from(config)
    n_parameters = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    #ipdb.set_trace()
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)


    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)


    optimizer2 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)


    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    dice_loss2 = losses.DiceLoss_confidence(num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0

    mask_thershold = args.mask_thershold*np.ones(args.num_classes)
    current_average_confidence_intra = 0.25*np.ones(args.num_classes)#.detach()
    average_confidence_intra = 0.25 * np.ones(args.num_classes)
    increase_intra = np.ones(args.num_classes)
    mask_class_ratio =  args.mask_class_ratio
    current_average_confidence_inter = 0.25*np.ones(args.num_classes)
    average_confidence_inter = 0.25 * np.ones(args.num_classes) #* np.ones(args.num_classes)
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        #torch.cuda.empty_cache()
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            alpfa = (iter_num/max_iterations)

            masked_volume_batch_intra, masked_volume_batch_inter, input_mask_intra, input_mask_inter, use_inter = mask_gen.mask_image_two_inter_intra_adaptive_flexmatch_spe_lianshi_conf(volume_batch[args.labeled_bs:], outputs_soft2[args.labeled_bs:], mask_thershold, mask_class_ratio, iter_num,average_confidence_inter, True,False)    

            masked_volume_batch_intra = torch.cat((volume_batch[:args.labeled_bs], masked_volume_batch_intra),0)
            masked_volume_batch_inter = torch.cat((volume_batch[:args.labeled_bs], masked_volume_batch_inter),0)
            input_mask_intra = torch.cat((input_mask_intra, input_mask_intra),0)
            input_mask_inter = torch.cat((input_mask_inter, input_mask_inter),0)


            outputs1_intra = model1(masked_volume_batch_intra)
            outputs_soft1_intra = torch.softmax(outputs1_intra, dim=1)

            outputs1_inter = model1(masked_volume_batch_inter)
            outputs_soft1_inter = torch.softmax(outputs1_inter, dim=1)


            consistency_weight = get_current_consistency_weight(
                iter_num // 150)

            loss1 = 0.5 * (ce_loss(outputs1_intra[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1_intra[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

                            
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))


            pseudo_outputs2 = torch.argmax(
                outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)
            

            
            pseudo_outputs1_intra = torch.argmax(
                outputs_soft1_intra[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1_intra = dice_loss(
                outputs_soft1_intra[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))#, input_mask_intra[args.labeled_bs:].squeeze(1))
            pseudo_supervision1_inter = dice_loss2(
                outputs_soft1_inter[args.labeled_bs:], pseudo_outputs2.unsqueeze(1), input_mask_inter[args.labeled_bs:].squeeze(1))
            #if use_inter:
            model1_loss_intra =  0.5*loss1 +  consistency_weight * pseudo_supervision1_intra 
            model1_loss_inter =  0.5*loss1 +  consistency_weight * pseudo_supervision1_inter
             #+ task_difficulty# + loss_reg
            #else:
                #model1_loss = loss1 + consistency_weight * pseudo_supervision1_intra
            #model1_loss_sup = loss1

            #optimizer1.zero_grad()
            #model1_loss_sup.backward(retain_graph=True)
            #optimizer2.step()
            #if iter_num < 200:
            #    loss = loss1
            #else:

            optimizer = PCGrad(optimizer1) 
            unsup_loss = [model1_loss_intra, model1_loss_inter] # a list of per-task losses
            assert len(unsup_loss) == 2
            optimizer.pc_backward(unsup_loss) # calculate the gradient can apply gradient modification
            #model1_loss_sup.backward()
            optimizer.step()  # apply gradient step

            #loss = model1_loss + model2_loss

            #optimizer2.zero_grad()
            #optimizer2.zero_grad()
            model1_loss = loss1 + model1_loss_intra + model1_loss_inter
            model2_loss = loss2 #+ consistency_weight * pseudo_supervision2


            # optimizer1.zero_grad()
            # model1_loss.backward()     
            # optimizer1.step()
            #volume_batch.grad.zero_()
            #masked_volume_batch.grad.zero_()
            #model1_loss_sup.backward()
            #data_grad = volume_batch[args.labeled_bs:].grad.data
            #print(masked_data_grad.unique())
            #print(masked_data_grad_stonger_aug.unique())
            #print((masked_data_grad_stonger_aug-masked_data_grad).unique())
            #print(data_grad)
            #ipdb.set_trace()
            #optimizer2.step()
            #optimizer2.step()
            #average_confidence = 
            #for class_i in

            #prediction_class_inter = torch.argmax(outputs_soft1_inter[args.labeled_bs:], dim=1)
            #ipdb.set_trace()
            #print("1:{}".format(torch.cuda.memory_allocated(0)))
            
            with torch.no_grad():
                #increase_inter = current_average_confidence_inter > average_confidence_inter
                #ipdb.set_trace()
                #print(average_confidence_intra, current_average_confidence_intra)
                #mask_thershold, mask_class_ratio = adjust_parameters_by_confidence(increase_intra, increase_inter, mask_thershold, mask_class_ratio, adjustment_rate)
                increase_intra, increase_intra_num, increase_inter, increase_inter_num = update_confidence(args, outputs_soft1_intra, input_mask_intra, outputs_soft1_inter, input_mask_inter, outputs_soft2, current_average_confidence_intra, increase_intra, average_confidence_intra, current_average_confidence_inter, average_confidence_inter)
                mask_thershold, mask_class_ratio = adjust_parameters_by_confidence2(args, increase_intra, increase_intra_num, increase_inter, increase_inter_num, mask_thershold, mask_class_ratio, alpfa)  
                #for class_i in all_existing_class:
                    #average_confidence_intra[class_i] = current_average_confidence_intra[class_i]
                #mask_thershold[0] = 0.75
                #average_confidence_inter = current_average_confidence_inter#.clone()
                #print(mask_thershold, mask_class_ratio)
                #print(average_confidence_intra, average_confidence_inter)
                update_ema_variables(model1, model2, args.ema_decay, iter_num)

                average_confidence_intra, average_confidence_inter = update_ema_average_confidence(current_average_confidence_intra, average_confidence_intra, current_average_confidence_inter, average_confidence_inter, args.ema_decay, iter_num)
                #if alpfa <= 0.05:
                #    mask_thershold, mask_class_ratio = 0.25*np.ones(args.num_classes), 0.25
                #print("avi:",average_confidence_intra, "aci:",average_confidence_inter)
            iter_num = iter_num + 1
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            if iter_num > 0 and iter_num % 50 == 0:
                logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item()))
                logging.info('use_inter : %d ' % use_inter)
                logging.info('mask_thre %f %f %f %f : mask_cls %f '% (mask_thershold[0].item(), mask_thershold[1].item(), mask_thershold[2].item(), mask_thershold[3].item(), mask_class_ratio.item()))
                logging.info('avi %f %f %f %f : aci %f' % (average_confidence_intra[0].item(), average_confidence_intra[1].item(), average_confidence_intra[2].item(), average_confidence_intra[3].item(), np.mean(average_confidence_inter).item()))
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1_intra, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 5000 and iter_num % 500 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                #ipdb.set_trace()
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_jaccard'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/model1_val_{}_asd'.format(class_i+1),
                                      metric_list[class_i, 3], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]
                mean_jaccard1 = np.mean(metric_list, axis=0)[1]
                mean_hd951 = np.mean(metric_list, axis=0)[2]
                mean_asd1 = np.mean(metric_list, axis=0)[3]
                writer.add_scalar('info/model1_val_mean_dice',
                                  performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_jaccard',
                                  mean_jaccard1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95',
                                  mean_hd951, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95',
                                  mean_asd1, iter_num)                        

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_jaccard : %f model1_mean_hd95 : %f model1_mean_asd : %f' % (iter_num, performance1, mean_jaccard1, mean_hd951, mean_asd1))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_jaccard'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/model1_val_{}_asd'.format(class_i+1),
                                      metric_list[class_i, 3], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]
                mean_jaccard2 = np.mean(metric_list, axis=0)[1]
                mean_hd952 = np.mean(metric_list, axis=0)[2]
                mean_asd2 = np.mean(metric_list, axis=0)[3]
                writer.add_scalar('info/model1_val_mean_dice',
                                  performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_jaccard',
                                  mean_jaccard1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95',
                                  mean_hd951, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95',
                                  mean_asd1, iter_num)                 

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_jaccard : %f model2_mean_hd95 : %f model2_mean_asd : %f' % (iter_num, performance2, mean_jaccard2, mean_hd952, mean_asd2))
                model2.train()

            if iter_num % 5000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_label_{}_adaptive_ins_{}_back_{}_size_{}_alpfa_{}_init_thershold_{}_init_class_ratio_{}/{}".format(
        args.exp, args.labeled_num, args.adjustment_rate_intra, args.adjustment_rate_inter, args.mask_size, args.consistency,args.mask_thershold, args.mask_class_ratio, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    #shutil.copytree('.', snapshot_path + '/code',
                    #shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)


