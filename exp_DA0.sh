python main_DA.py --gpu_id=0 --fold_num 0 1 2 3 --bAttention_Fixation 0 --bFeatureMap_Regularization 0 --fe_lambda 10 --seg_lambda 1 --fm_lambda 0.01 --exp_name DA_fe10_seg1_fm0.01_FF
python main_DA.py --gpu_id=0 --fold_num 0 1 2 3 --bAttention_Fixation 0 --bFeatureMap_Regularization 0 --fe_lambda 1 --seg_lambda 0.1 --fm_lambda 0.01 --exp_name DA_fe1_seg0.1_fm0.01_FF
python main_DA.py --gpu_id=0 --fold_num 0 1 2 3 --bAttention_Fixation 1 --bFeatureMap_Regularization 0 --fe_lambda 10 --seg_lambda 1 --fm_lambda 0.01 --exp_name DA_fe10_seg1_fm0.01_TF
