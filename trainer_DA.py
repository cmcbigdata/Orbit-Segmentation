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
    def __init__(self, epochs, gpu, batch_size, image_size, num_domain, discriminator_learning_rate, generator_learning_rate, bAttention_Fixation, bFeatureMap_Regularization, fe_lambda, seg_lambda, fm_lambda, output_dir, pretrained_model_dir, src_train_dataloader, tg_train_dataloader, tg_test_dataloader, writer):
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
        self.num_domain = num_domain

        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_learning_rate = generator_learning_rate

        self.bAttention_Fixation = bAttention_Fixation 
        self.bFeatureMap_Regularization = bFeatureMap_Regularization       
        self.fe_lambda = fe_lambda
        self.seg_lambda = seg_lambda    
        self.fm_lambda = fm_lambda    

        self.output_dir = output_dir
        self.pretrained_model_dir = pretrained_model_dir

        self.src_train_dataloader = src_train_dataloader
        self.tg_train_dataloader = tg_train_dataloader
        self.tg_test_dataloader = tg_test_dataloader

        self.writer = writer
    
    def fix_attention_parameters(self, model):
        for param in model.Attention3.parameters():
            param.requires_grad = False
        for param in model.Attention2.parameters():
            param.requires_grad = False
        for param in model.Attention1.parameters():
            param.requires_grad = False
        
    
    def train(self):
        from model.DA_model import DeepSequentialNet, Domain_Discriminator
        src_sqNet = DeepSequentialNet(self.image_size, self.device).to(self.device)
        src_sqNet.load_state_dict(torch.load(self.pretrained_model_dir))   ### transfer learning
        src_sqNet.eval()
        src_sqNet.requires_grad_(False)

        tg_sqNet = DeepSequentialNet(self.image_size, self.device).to(self.device)
        tg_sqNet.load_state_dict(torch.load(self.pretrained_model_dir))

        domain_Dnet = Domain_Discriminator(512, self.num_domain).to(self.device)
        domain_Dnet.apply(weights_init)
        print("###### Domain adaptation ######")

            
        total_param = sum(p.numel() for p in tg_sqNet.parameters())
        train_param = sum(p.numel() for p in tg_sqNet.parameters() if p.requires_grad)
      
        
        optimizer_sqNet = optim.Adam(tg_sqNet.parameters(), lr=self.generator_learning_rate, betas=(0.9, 0.999))
        optimizer_domainNet = optim.Adam(domain_Dnet.parameters(), lr=self.discriminator_learning_rate, betas=(0.9, 0.999))

        DICE_criterion = DiceLoss().to(self.device)
        CE_criterion = nn.CrossEntropyLoss().to(self.device)

        
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
            elif epoch == 50:
                result_print_step = 1000
            
            start_time_epoch = time.time()
            
            for src_data, tg_data in zip(self.src_train_dataloader, self.tg_train_dataloader):
                ### data
                src_seq_vols, _ = src_data
                tg_seq_vols, tg_masks = tg_data
                src_seq_vols = Variable(src_seq_vols).float().to(self.device)
                tg_seq_vols = Variable(tg_seq_vols).float().to(self.device)
                tg_masks = Variable(tg_masks).float().to(self.device)

                src_labels = Variable(torch.LongTensor(src_seq_vols.shape[0]).fill_(0)).to(self.device)
                tg_labels = Variable(torch.LongTensor(tg_seq_vols.shape[0]).fill_(1)).to(self.device)

                tg_sqNet.train()
                tg_sqNet.requires_grad_(True)
                ### fix attention parameter 
                if self.bAttention_Fixation:
                    self.fix_attention_parameters(tg_sqNet)

                ### prediction
                src_pred_features = src_sqNet(src_seq_vols, 'feature_extractor')
                tg_pred_features = tg_sqNet(tg_seq_vols, 'feature_extractor')
                
                ########################
                ## train domain discriminator #
                ########################
                domain_Dnet.train()
                domain_Dnet.requires_grad_(True)
                src_domain_logit1 = domain_Dnet(src_pred_features.detach())
                tg_domain_logit1 = domain_Dnet(tg_pred_features.detach())

                err_domainD = CE_criterion(src_domain_logit1, src_labels) + CE_criterion(tg_domain_logit1, tg_labels)
                
                domain_Dnet.zero_grad()
                err_domainD.backward()
                optimizer_domainNet.step()
                domain_Dnet.requires_grad_(False)

                ###############
                # train sqNet #
                ###############
                ### feature extractor (encoder part)
                tg_pred_features = tg_sqNet(tg_seq_vols, 'feature_extractor')
                tg_domain_logit2 = domain_Dnet(tg_pred_features)
                src_labels2 = Variable(torch.LongTensor(tg_seq_vols.shape[0]).fill_(0)).to(self.device)
                err_sqNet_domain = CE_criterion(tg_domain_logit2, src_labels2)


                tg_pred_masks = tg_sqNet(tg_seq_vols, 'segmentation')
                
                cnt_nervePixel = tg_masks.nonzero().size(0) + 1           # mask = 1 (nodule)
                cnt_allPixel = tg_masks.numel()                           # all mask pixel
                dice_loss_weight = cnt_allPixel/cnt_nervePixel     
                       
                dice_score = compute_dice_coeff_train(tg_pred_masks, tg_masks)
                err_sqNet_seg = dice_loss_weight + DICE_criterion(tg_pred_masks, tg_masks, dice_loss_weight) 

                if self.bFeatureMap_Regularization:
                    tg_pred_features_from_src_fe = src_sqNet(tg_seq_vols, 'feature_extractor')
                    err_feature_map = self.extractor_att_fea_map(tg_pred_features_from_src_fe, tg_pred_features)

                    err_sqNet = self.fe_lambda * err_sqNet_domain + self.seg_lambda * err_sqNet_seg + self.fm_lambda * err_feature_map
                else:
                    err_sqNet = self.fe_lambda * err_sqNet_domain + self.seg_lambda * err_sqNet_seg 
                
                
                tg_sqNet.zero_grad()
                err_sqNet.backward()
                optimizer_sqNet.step()
                tg_sqNet.requires_grad_(False)

                

                if total_step % loss_print_step == 0:
                    end_time_step = time.time() 
                    print('[%d / %d]   time : %.2fs' % (epoch, self.epochs, end_time_step - start_time_step))     
                    
                    self.writer.add_scalar('train/domain disciminator loss', err_domainD.item(), total_step)
                    self.writer.add_scalar('train/feature extractor loss', err_sqNet_domain.item(), total_step)
                    self.writer.add_scalar('train/step_dice_loss', err_sqNet_seg.item(), total_step)
                    self.writer.add_scalar('train/generator loss', err_sqNet.item(), total_step)
                    self.writer.add_scalar('train/step_dice_coeff', dice_score.item(), total_step)
                    if self.bFeatureMap_Regularization:
                        self.writer.add_scalar('train/err_feature_map', err_feature_map.item(), total_step)
                    

                    if total_step % result_print_step == 0:
                        with torch.no_grad():
                            tg_seq_vols = tg_seq_vols.cpu().numpy()
                            tg_seq_vols = np.transpose(tg_seq_vols, (0, 2, 3, 4, 1))
                            tg_seq_vols = np.squeeze(tg_seq_vols)

                            tg_masks = tg_masks.cpu().numpy()
                            tg_masks = np.squeeze(tg_masks)

                            tg_pred_masks = tg_pred_masks.cpu().numpy()
                            tg_pred_masks = np.squeeze(tg_pred_masks)

                            result_imgs = np.array([])
                            for train_result_idx, (gt, m, pred_m) in enumerate(zip(tg_seq_vols, tg_masks, tg_pred_masks)):
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
            tg_sqNet.eval()
            tg_sqNet.requires_grad_(False)

            with torch.no_grad():
                total_dice_coeff = []
                total_w_dice_coeff = []
                total_seg_loss = []
                total_vs = []
                total_num_test_batch = []
                
                for test_idx, data in enumerate(self.tg_test_dataloader):
                    test_seq_vols, test_masks = data
                    test_seq_vols = Variable(test_seq_vols).float().to(self.device)
                    test_masks = Variable(test_masks).float().to(self.device)
                                                            
                    test_pred_masks = tg_sqNet(test_seq_vols, 'segmentation')
                    
                    cnt_nodulePixel = test_masks.nonzero().size(0) + 1          # mask = 1 (nodule)
                    cnt_allPixel = test_masks.numel()                           # all mask pixel
                    loss_weight = cnt_allPixel/cnt_nodulePixel
                    dice_coeff = compute_dice_coeff_test(test_pred_masks, test_masks)
                    w_dice_coeff = compute_dice_coeff_train(test_pred_masks, test_masks)
                    seg_loss = loss_weight + DICE_criterion(test_pred_masks, test_masks, loss_weight)

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
                        torch.save(tg_sqNet.state_dict(), '%s/tg_sqNet_best.pth' % (self.model_dir))

                    elif worst_dice_score > total_dice_coeff:
                        worst_dice_score = total_dice_coeff

                    if best_volume_metric < total_vs:
                        best_volume_metric = total_vs

                    elif worst_volume_metric > total_vs:
                        worst_volume_metric = total_vs
                    
                    av_dice_score += total_dice_coeff
                    av_volume_metric += total_vs
                
           

        torch.save(tg_sqNet.state_dict(), '%s/tg_sqNet_final.pth' % (self.model_dir))                            #########이거 지워야 함
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



