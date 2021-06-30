python main.py --gpu_id=1 --fold_num 2 3 --bAttention 0 --bTransfer_learning 0 --exp_name sensor3d
python main.py --gpu_id=1 --fold_num 2 3 --bAttention 1 --bTransfer_learning 0 --exp_name sensor3d_attention
python main.py --gpu_id=1 --fold_num 2 3 --bAttention 1 --bTransfer_learning 1 --exp_name sensor3d_attention_transfer