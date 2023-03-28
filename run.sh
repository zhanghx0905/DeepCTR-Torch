CUDA_VISIBLE_DEVICES=0 python main.py --save_path ckpt/raw_37.pth 
CUDA_VISIBLE_DEVICES=1 python main.py --dense_embedding --save_path ckpt/dense_embedding_37.pth 

CUDA_VISIBLE_DEVICES=2 python main.py --save_path ckpt/no_dnn_37.pth --not_use_dnn 
CUDA_VISIBLE_DEVICES=3 python main.py --save_path ckpt/no_dnn_dense_embedding_37.pth  --not_use_dnn --dense_embedding

CUDA_VISIBLE_DEVICES=0 python main.py --save_path ckpt/no_cin_dnn.pth --not_use_dnn --not_use_cin
