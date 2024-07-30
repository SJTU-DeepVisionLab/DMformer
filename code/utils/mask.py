import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import numpy as np

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class BlockMaskGenerator:

    def __init__(self, mask_ratio=0.4, mask_ratio_stronger=0.7, mask_block_size=14, noise_ratio=0.5):
        self.mask_ratio = mask_ratio
        self.mask_ratio_stronger = mask_ratio_stronger
        self.mask_block_size = mask_block_size
        self.noise_ratio = noise_ratio


    @torch.no_grad()
    def generate_mask(self, imgs, outputs_softmax, mask_thershold, mask_class_ratio, iter_num, average_class_ratio, random1=True, random2=True, class_nums=4):
        #ipdb.set_trace()):
        B, C, H, W = imgs.shape
        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)

        outputs_softmax = resize(outputs_softmax, size=(mshape[2],mshape[3]))

        use_inter = False
        input_mask_intra_all = torch.zeros(mshape, device=imgs.device) 
        input_mask_inter_all = torch.zeros(mshape, device=imgs.device)  
        all_existing_class = torch.unique(torch.argmax(outputs_softmax, dim=1))
        prediction_class = torch.argmax(outputs_softmax, dim=1).view(B,-1)
        class_i_conf = torch.zeros(class_nums)
        for class_i in all_existing_class:
            class_i_conf[class_i] = average_class_ratio[class_i]

        len_unique = len(all_existing_class)
        if len(all_existing_class) > 1:
            if random1:
                rand_list = torch.randperm(len_unique)

                check_list = rand_list[:max(1,int(mask_class_ratio * len_unique))]
            else:
                prob = class_i_conf/class_i_conf.sum()
                check_list = torch.multinomial(prob, max(1,int(mask_class_ratio * len_unique)), replacement=False) #sort_inter_class[:max(1,int(mask_class_ratio * len_unique))]

            inference_inter_class = check_list.cuda()

            if int(mask_class_ratio * len_unique) >= 1:
                use_inter = True
            else:
                use_inter = False

        else:
            inference_inter_class = -1 + torch.zeros(1).cuda()
            use_inter = False
        for class_i in all_existing_class:
            number_belong_class_i = torch.sum((prediction_class == class_i), dim=1).type(torch.int64)#.cuda()

            class_not_region = (prediction_class != class_i)

            mask_pos = (number_belong_class_i * mask_thershold[class_i]).type(torch.int64)


            if random2:
                input_mask_intra = torch.rand(prediction_class.size(), device=imgs.device) 
                input_mask_intra[class_not_region] = -1 
                input_mask_intra_order = torch.sort(input_mask_intra.view(B,-1), descending=True)[0]
                mask_value_intra = input_mask_intra_order.gather(1, mask_pos.unsqueeze(1))
                input_mask_intra_final = ((input_mask_intra <= mask_value_intra) & (input_mask_intra >=0)).float()#.unsqueeze(1) 

            else:
                entropy = torch.max(outputs_softmax,dim=1)[0].view(B,-1)#-torch.sum(torch.log2(outputs_softmax) * outputs_softmax, dim=1).view(B,-1) 
                entropy[class_not_region] = -99
                confidence = torch.softmax(entropy, dim=1)
                entropy_order = torch.multinomial(confidence, entropy.size(1), replacement=False) 
                input_mask_intra_final = torch.zeros_like(entropy).cuda() 
                for batch_i in range(mask_pos.size(0)):
                    indices = entropy_order[batch_i][:mask_pos[batch_i]]
                    input_mask_intra_final[batch_i][indices] = 1
                    input_mask_intra_final[batch_i] = 1 - input_mask_intra_final[batch_i]
                input_mask_intra_final = ((input_mask_intra_final > 0) & (entropy >=0)).float() 

                entropy = torch.max(outputs_softmax,dim=1)[0].view(B,-1)#-torch.sum(torch.log2(outputs_softmax) * outputs_softmax, dim=1).view(B,-1) 
                entropy[class_not_region] = -99
                confidence = torch.softmax(entropy, dim=1)
                entropy_order = torch.multinomial(confidence, entropy.size(1), replacement=False) 
                input_mask_intra_final = torch.zeros_like(entropy).cuda() 
                for batch_i in range(mask_pos.size(0)):
                    indices = entropy_order[batch_i][:mask_pos[batch_i]]
                    input_mask_intra_final[batch_i][indices] = 1
                    input_mask_intra_final[batch_i] = 1 - input_mask_intra_final[batch_i]
                input_mask_intra_final = ((input_mask_intra_final > 0) & (entropy >=0)).float() 

            if class_i in inference_inter_class:
                input_mask_inter_final = torch.zeros_like(input_mask_intra_final).cuda()#(input_mask_intra >=0)

            else:
                input_mask_inter_final = input_mask_intra_final 
            input_mask_inter_all += input_mask_inter_final.reshape(B,1, mshape[2],mshape[3])          
            input_mask_intra_all += input_mask_intra_final.reshape(B,1, mshape[2],mshape[3])
        input_mask_intra_all = resize(input_mask_intra_all, size=(H, W))
        input_mask_inter_all = resize(input_mask_inter_all, size=(H, W))
        input_mask_intra_all[input_mask_intra_all>1] = 1
        input_mask_inter_all[input_mask_inter_all>1] = 1
        return input_mask_intra_all, input_mask_inter_all, use_inter

    @torch.no_grad()
    def mask_image_two_inter_intra_adaptive_flexmatch_spe_lianshi_conf(self, imgs, outputs_softmax, mask_thershold, mask_class_ratio, iter_num, average_class_ratio, random1=True, random2=True, class_num=4):
        input_mask_intra, input_mask_inter, use_inter = self.generate_mask(imgs, outputs_softmax, mask_thershold, mask_class_ratio, iter_num, average_class_ratio, random1, random2, class_num)
        img_intra = imgs * input_mask_intra
        img_inter = imgs * input_mask_inter
        return img_intra, img_inter, input_mask_intra, input_mask_inter, use_inter


    
    