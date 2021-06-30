import os
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import random
import torchvision
import matplotlib.pyplot as plt
from medpy.io import load, save
import cv2

from model.loss import DiceLoss

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)

def compute_dice_coeff_train(pred, gt, smooth=1):
    pred = pred.view(-1)
    gt = gt.view(-1)
    
    intersection = (pred * gt).sum()           
    dice = (2.0*intersection + smooth)/(pred.sum() + gt.sum() + smooth)
        
    return dice

def compute_dice_coeff_test(pred, gt, smooth=1):
    pred = (pred > 0.5).float()
    pred = pred.view(-1)
    gt = gt.view(-1)
    
    intersection = (pred * gt).sum()           
    dice = (2.0*intersection + smooth)/(pred.sum() + gt.sum() + smooth)
        
    return dice

def compute_vs(pred, gt):
    pred = (pred > 0.5).float()
    
    single_vs = []
    for batch in range(pred.shape[0]):
        confusion_vector = pred[batch, :, :, :] / gt[batch, :, :, :]
        
        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        # true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        vs = 1 - abs(false_negatives-false_positives) / (2*true_positives + false_positives + false_negatives + 1e-4)
        single_vs.append(vs)
    
    total_sum_vs = sum(single_vs)
    return total_sum_vs



class sequentialSegTrainer(object):
    def __init__(self, epochs, gpu, batch_size, image_size, learning_rate, output_dir, bAttention, bTransfer_learning, pretrained_model_dir, train_dataloader, test_dataloader, writer):
        self.model_dir = os.path.join(output_dir, 'model')
        self.train_result_dir = os.path.join(output_dir, 'result', 'train')
        self.test_result_dir = os.path.join(output_dir, 'result', 'test')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.train_result_dir)
            os.makedirs(self.test_result_dir)

        self.epochs = epochs
        self.device = torch.device("cuda:%s" % gpu)
        self.batch_size = batch_size
        self.image_size = image_size

        self.learning_rate = learning_rate

        self.output_dir = output_dir
        self.bAttention = bAttention
        self.bTransfer_learning = bTransfer_learning
        self.pretrained_model_dir = pretrained_model_dir

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.writer = writer
    
    def fix_attention_parameters(self, model):
        for param in model.Attention3.parameters():
            param.requires_grad = False
        for param in model.Attention2.parameters():
            param.requires_grad = False
        for param in model.Attention1.parameters():
            param.requires_grad = False
        
    
    def train(self):
        if self.bAttention and not(self.bTransfer_learning):
            from model.attention_model import DeepSequentialNet
            sqNet = DeepSequentialNet(self.image_size, self.device).to(self.device)
            sqNet.apply(weights_init)
            print("###### Sensor3D + Attention model ######")
            
        elif not(self.bAttention) and not(self.bTransfer_learning):
            from model.sensor3d_model import DeepSequentialNet
            sqNet = DeepSequentialNet(self.image_size, self.device).to(self.device)
            sqNet.apply(weights_init)
            print("###### Sensor3D model ######")
        
        elif self.bAttention and self.bTransfer_learning:
            from model.attention_model import DeepSequentialNet
            sqNet = DeepSequentialNet(self.image_size, self.device).to(self.device)
            sqNet.load_state_dict(torch.load(self.pretrained_model_dir, map_location=self.device))
            print("###### Sensor3D + Attention model + transfer learning ######")

            
        total_param = sum(p.numel() for p in sqNet.parameters())
        train_param = sum(p.numel() for p in sqNet.parameters() if p.requires_grad)

        
        
        optimizer = optim.Adam(sqNet.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        criterion = DiceLoss().to(self.device)

        
        start_t = time.time()

        total_step = 0
        result_print_step=10
        loss_print_step = 10
        start_time_step = time.time()
        test_img_export_interval = 5
        
        best_dice_score = 0
        worst_dice_score = 0 

        for epoch in range(self.epochs):
            if epoch == 10:
                result_print_step = 100
                test_img_export_interval = 20
            elif epoch == 100:
                result_print_step = 1000
            
            start_time_epoch = time.time()
            
            for data in self.train_dataloader:
                seq_vols, masks = data

                seq_vols = Variable(seq_vols).float().to(self.device)
                masks = Variable(masks).float().to(self.device)

                cnt_nodulePixel = masks.nonzero(as_tuple=False).size(0) + 1          # mask = 1 (nodule)
                cnt_allPixel = masks.numel()                           # all mask pixel
                loss_weight = cnt_allPixel/cnt_nodulePixel

                

                sqNet.train()
                sqNet.requires_grad_(True)
                pred_masks = sqNet(seq_vols)
                
                dice_coeff = compute_dice_coeff_train(pred_masks, masks)
                loss = loss_weight + criterion(pred_masks, masks, loss_weight)
                
                sqNet.zero_grad()
                loss.backward()
                optimizer.step()
                sqNet.requires_grad_(False)

                

                if total_step % loss_print_step == 0:
                    end_time_step = time.time() 
                    print('[%d / %d]   time : %.2fs' % (epoch, self.epochs, end_time_step - start_time_step))     
                    
                    self.writer.add_scalar('train/step_dice_coeff', dice_coeff.item(), total_step)
                    self.writer.add_scalar('train/step_dice_loss', loss.item(), total_step)
                
                    

                    if total_step % result_print_step == 0:
                        with torch.no_grad():
                            seq_vols = seq_vols.cpu().numpy()
                            seq_vols = np.transpose(seq_vols, (0, 2, 3, 4, 1))
                            seq_vols = np.squeeze(seq_vols)

                            masks = masks.cpu().numpy()
                            masks = np.squeeze(masks)

                            pred_masks = pred_masks.cpu().numpy()
                            pred_masks = np.squeeze(pred_masks)

                            result_imgs = np.array([])
                            for train_result_idx, (gt, m, pred_m) in enumerate(zip(seq_vols, masks, pred_masks)):
                                # gt = gt*255
                                # m = m*255
                                # pred_m = pred_m*255

                                # result_img = np.concatenate((gt[:,:,1], m, pred_m), 1)
                                                                
                                pred_m = np.where(pred_m>0.5, 1, 0)

                                b_gtmask = np.zeros([64,64,3])
                                b_gtmask[:,:,0] = 255
                                b_predmask = np.zeros([64,64,3])
                                b_predmask[:,:,2] = 255

                                gt = cv2.cvtColor(gt[:,:,1]*255, cv2.COLOR_GRAY2RGB) 
                                b_gtmask[:,:,0] = m*b_gtmask[:,:,0]
                                b_predmask[:,:,2] = pred_m*b_predmask[:,:,2]
                                
                                temp_result_img1 = cv2.addWeighted(gt.astype(np.uint8), 0.7, b_gtmask.astype(np.uint8), 0.3, 0)
                                temp_result_img2 = cv2.addWeighted(temp_result_img1, 0.7, b_predmask.astype(np.uint8), 0.3, 0)
                                
                                result_img = np.concatenate((gt, temp_result_img1, temp_result_img2), 1)

                                if train_result_idx == 0:
                                    result_imgs = result_img
                                else:
                                    result_imgs = np.concatenate((result_imgs, result_img), 0)
                                
                                if train_result_idx == 6:
                                    break
                            cv2.imwrite(self.train_result_dir + '/epoch_' + str(epoch) + '_step' + str(total_step) + '.png', result_imgs)
                
                    start_time_step = time.time()    
                total_step += 1

            end_time_epoch = time.time()
            print('[%d / %d - %d step] training time : %.5fs ' % (epoch, self.epochs, total_step, end_time_epoch-start_time_epoch))

            ##########################################################
            ##########################################################
            sqNet.eval()
            sqNet.requires_grad_(False)

            with torch.no_grad():
                total_dice_coeff = []
                total_w_dice_coeff = []
                total_seg_loss = []
                total_vs = []
                total_num_test_batch = []
                
                for test_idx, data in enumerate(self.test_dataloader):
                    test_seq_vols, test_masks = data

                    test_seq_vols = Variable(test_seq_vols).float().to(self.device)
                    test_masks = Variable(test_masks).float().to(self.device)
                    
                    test_pred_masks = sqNet(test_seq_vols)

                    cnt_nodulePixel = test_masks.nonzero(as_tuple=False).size(0) + 1          # mask = 1 (nodule)
                    cnt_allPixel = test_masks.numel()                           # all mask pixel
                    loss_weight = cnt_allPixel/cnt_nodulePixel
                    w_dice_coeff = compute_dice_coeff_train(test_pred_masks, test_masks) 
                    dice_coeff = compute_dice_coeff_test(test_pred_masks, test_masks)                     
                    seg_loss = loss_weight + criterion(test_pred_masks, test_masks, loss_weight)   
                    vs = compute_vs(test_pred_masks, test_masks)
                    

                    #################################################
                    #################################################
                    batch_size = test_seq_vols.shape[0]
                    total_dice_coeff.append(dice_coeff.item()*batch_size)
                    total_w_dice_coeff.append(w_dice_coeff.item()*batch_size)
                    total_seg_loss.append(seg_loss.item()*batch_size)
                    total_vs.append(vs)
                    total_num_test_batch.append(batch_size)
                    #################################################
                    #################################################

                    if test_idx == 0 and epoch%test_img_export_interval == 0:
                        test_pred_masks = test_pred_masks.cpu().numpy()
                        test_pred_masks = np.squeeze(test_pred_masks)

                        test_masks = test_masks.cpu().numpy()
                        test_masks = np.squeeze(test_masks)

                        test_seq_vols = test_seq_vols.cpu().numpy()
                        test_seq_vols = np.transpose(test_seq_vols, (0, 2, 3, 4, 1))
                        test_seq_vols = np.squeeze(test_seq_vols)
                        
                        result_imgs = np.array([])
                       
                        for idx, (gt, m, pred_m) in enumerate(zip(test_seq_vols, test_masks, test_pred_masks)):
                            # gt = gt*255
                            # m = m*255
                            # pred_m = pred_m*255

                            # result_img = np.concatenate((gt[:,:,1], m, pred_m), 1)
                            pred_m = np.where(pred_m>0.5, 1, 0)

                            b_gtmask = np.zeros([64,64,3])
                            b_gtmask[:,:,0] = 255
                            b_predmask = np.zeros([64,64,3])
                            b_predmask[:,:,2] = 255

                            gt = cv2.cvtColor(gt[:,:,1]*255, cv2.COLOR_GRAY2RGB) 
                            b_gtmask[:,:,0] = m*b_gtmask[:,:,0]
                            b_predmask[:,:,2] = pred_m*b_predmask[:,:,2]
                            
                            temp_result_img1 = cv2.addWeighted(gt.astype(np.uint8), 0.7, b_gtmask.astype(np.uint8), 0.3, 0)
                            temp_result_img2 = cv2.addWeighted(temp_result_img1, 0.7, b_predmask.astype(np.uint8), 0.3, 0)

                            result_img = np.concatenate((gt, temp_result_img1, temp_result_img2), 1)

                            if idx == 0:
                                result_imgs = result_img
                            else:
                                result_imgs = np.concatenate((result_imgs, result_img), 0)

                            
                            if idx == 6:
                                break
                        cv2.imwrite(self.test_result_dir + '/epoch_' + str(epoch) + '_step' + str(total_step) + '.png', result_imgs)
                

                total_dice_coeff = sum(total_dice_coeff) / sum(total_num_test_batch)
                total_w_dice_coeff = sum(total_w_dice_coeff) / sum(total_num_test_batch)
                total_seg_loss = sum(total_seg_loss) / sum(total_num_test_batch)
                total_vs = sum(total_vs) / sum(total_num_test_batch)
                

                self.writer.add_scalar('test/dice coeff', total_dice_coeff, epoch)
                self.writer.add_scalar('test/wrong dice coeff', total_w_dice_coeff, epoch)
                self.writer.add_scalar('test/segmentation loss', total_seg_loss, epoch)
                
                self.writer.add_scalar('test/volume similarity', total_vs, epoch)

                if epoch == self.epochs - 50:
                    best_dice_score = total_dice_coeff
                    worst_dice_score = total_dice_coeff
                    av_dice_score = total_dice_coeff

                    best_volume_metric = total_vs
                    worst_volume_metric = total_vs
                    av_volume_metric = total_vs   

                elif epoch > self.epochs - 50:
                    if best_dice_score < total_dice_coeff:
                        best_dice_score = total_dice_coeff
                        torch.save(sqNet.state_dict(), '%s/sqNet_best.pth' % (self.model_dir))

                    elif worst_dice_score > total_dice_coeff:
                        worst_dice_score = total_dice_coeff

                    if best_volume_metric < total_vs:
                        best_volume_metric = total_vs

                    elif worst_volume_metric > total_vs:
                        worst_volume_metric = total_vs
                    
                    av_dice_score += total_dice_coeff
                    av_volume_metric += total_vs
                
           

        torch.save(sqNet.state_dict(), '%s/sqNet_final.pth' % (self.model_dir))                            #########이거 지워야 함
        print("best_dice_score  : ", best_dice_score)
        print("worst_dice_score  : ", worst_dice_score)
        end_t = time.time()

        f = open(self.model_dir + '/best_worst_dice_score.txt', 'w')
        f.write("best_dice_score : %.5f \n" % (best_dice_score))
        f.write("worst_dice_score : %.5f \n" % (worst_dice_score))
        f.write('Average_dice_scroe : %.5f \n' %(av_dice_score/50))
        f.write("best_Volumetric_Similarity : %.5f \n" % (best_volume_metric))
        f.write("worst_Volumetric_Similarity : %.5f \n" % (worst_volume_metric))
        f.write('Average_Volumetric_Similarity : %.5f \n' %(av_volume_metric/50))
        f.write('Total Param : %i \n' %(total_param))
        f.write('Train Param : %i \n' %(train_param))
        f.write('Total Training Time : %.5f \n' % ( end_t - start_t))
        f.close()

            
    def flatten_outputs(self, fea):
        return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2], fea.shape[3]*fea.shape[4]))
        
    def extractor_att_fea_map(self, fm_src, fm_tgt):
        fea_loss = torch.tensor(0.).to(self.device)
        
        b, s, c, h, w = fm_src.shape
        fm_src = self.flatten_outputs(fm_src)
        fm_tgt = self.flatten_outputs(fm_tgt)

        distance = torch.norm(fm_tgt - fm_src.detach(), 2, 3)
        distance = distance ** 2 / (h * w)
        fea_loss += torch.sum(distance)
        return fea_loss      



