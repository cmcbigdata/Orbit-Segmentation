import os
import random
import argparse
import time
import datetime
import dateutil.tz
import torch
import pandas as pd
import numpy as np

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='0', help='GPU number')
    parser.add_argument("--num_workers", type=int, default=4, help='worker number')
    parser.add_argument("--epochs", type=int, default=500, help="epochs")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    
    parser.add_argument('--discriminator_learning_rate', type=float, default=1e-4, help='discriminator learning rate')     
    parser.add_argument('--generator_learning_rate', type=float, default=5e-5, help='generator learning rate')     
    parser.add_argument('--total_fold_num', type=int, default=4, help='num of total fold')
    parser.add_argument('--fold_num', type=int, nargs='+', required=True, help='num of target fold')
    
    parser.add_argument("--num_sequence", type=int, default=3, help='number of sequence')
    parser.add_argument("--image_size", type=int, default=64, help='image size')
    parser.add_argument("--num_domain", type=int, default=2, help='number of domain')

    parser.add_argument('--bAttention_Fixation', type=int, default=0, help='bAttention_Fixation')    
    parser.add_argument('--bFeatureMap_Regularization', type=int, default=0, help='bFeatureMap_Regularization')     
    
    parser.add_argument('--fe_lambda', type=float, default=10, help='fe_lambda')
    parser.add_argument('--seg_lambda', type=float, default=1, help='seg_lambda')     
    parser.add_argument('--fm_lambda', type=float, default=0.01, help='fm_lambda')     

    parser.add_argument("--exp_name", type=str, default='test', help='experiment name')
    parser.add_argument("--pretrained_model_path", type=str, default='pretrained_model/nodule_sensor3d_attention_final.pth', help='pretrained_model_path')
    parser.add_argument("--tg_dataset_path", type=str, default='/home/ubuntu/data/Workspace/Wonseo/Orbit_Seg_bySJ/FINAL/CMC_seq/', help='tg_dataset_path')
    parser.add_argument("--src_dataset_path", type=str, default='/home/ubuntu/data/Workspace/Wonseo/Orbit_Seg_bySJ/LIDC_seq/', help='src_dataset_path')

    
    opt = parser.parse_args()
    print(opt)
    

    from tg_dataset import TargetDataset
    tg_train_dataset = []
    tg_test_dataset = []
    for fold in range(opt.total_fold_num):
        tg_train_dataset.append(TargetDataset(opt.tg_dataset_path, 'train', fold, opt.total_fold_num))
        tg_test_dataset.append(TargetDataset(opt.tg_dataset_path,'test', fold, opt.total_fold_num))
        print(fold, tg_train_dataset[fold].get_num_patient(), len(tg_train_dataset[fold]), tg_test_dataset[fold].get_num_patient(), len(tg_test_dataset[fold]))
 
    from src_dataset import SourceDataset
    src_train_dataset = []
    src_total_fold_num = 10
    for fold in range(opt.total_fold_num):
        src_train_dataset.append(SourceDataset(opt.src_dataset_path, 'train', tg_train_dataset[fold].get_num_patient() * 4, fold, src_total_fold_num))        
        
        while len(src_train_dataset[fold]) < len(tg_train_dataset[fold]):
            src_train_dataset.pop()
            src_train_dataset.append(SourceDataset(opt.src_dataset_path, 'train', tg_train_dataset[fold].get_num_patient() * 4, fold, src_total_fold_num))
        
        print(fold, tg_train_dataset[fold].get_num_patient(), len(tg_train_dataset[fold]), len(src_train_dataset[fold]))
        

    for fold in opt.fold_num:
        print(fold, '-th fold ::: training start    tg_train/tg_test/src_train')
        print(len(tg_train_dataset[fold]), len(tg_test_dataset[fold]), len(src_train_dataset[fold]))
        
        src_train_dataloader = torch.utils.data.DataLoader(src_train_dataset[fold], batch_size=opt.batch_size, drop_last=False, shuffle=True, num_workers=opt.num_workers)

        tg_train_dataloader = torch.utils.data.DataLoader(tg_train_dataset[fold], batch_size=opt.batch_size, drop_last=False, shuffle=True, num_workers=opt.num_workers)
        tg_test_dataloader = torch.utils.data.DataLoader(tg_test_dataset[fold], batch_size=opt.batch_size, drop_last=False, shuffle=True, num_workers=opt.num_workers)
        
        fold_exp_name = str(fold) + 'th_fold_' + opt.exp_name
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%m_%d_%H_%M')
        

        from torch.utils.tensorboard import SummaryWriter
        output_dir = './experiments_DA/%s_%s' % (timestamp, fold_exp_name)
        writer_path = './log/%s_%s' % (timestamp, fold_exp_name)
        os.makedirs(writer_path, exist_ok=True)   
        writer = SummaryWriter(writer_path)
        

        from trainer_DA import sequentialSegTrainer as trainer
        algo = trainer(epochs= opt.epochs, 
                        gpu= opt.gpu_id, 
                        batch_size= opt.batch_size, 
                        image_size= opt.image_size, 
                        num_domain= opt.num_domain, 
                        discriminator_learning_rate= opt.discriminator_learning_rate, 
                        generator_learning_rate= opt.generator_learning_rate, 
                        bAttention_Fixation= opt.bAttention_Fixation, 
                        bFeatureMap_Regularization= opt.bFeatureMap_Regularization, 
                        fe_lambda= opt.fe_lambda,
                        seg_lambda= opt.seg_lambda, 
                        fm_lambda= opt.fm_lambda, 
                        output_dir= output_dir, 
                        pretrained_model_dir=  opt.pretrained_model_path, 
                        src_train_dataloader= src_train_dataloader, 
                        tg_train_dataloader= tg_train_dataloader, 
                        tg_test_dataloader= tg_test_dataloader, 
                        writer= writer)


        start_t = time.time()
        algo.train()
        end_t = time.time()

        print(fold, '-th fold ::: total time for training: ', end_t - start_t)