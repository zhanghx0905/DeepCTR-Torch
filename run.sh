python main.py --save_path ckpt/raw_37.pth 
python main.py --dense_embedding --save_path ckpt/dense_embedding_37.pth 

python main.py --save_path ckpt/no_dnn_37.pth --not_use_dnn 
python main.py --save_path ckpt/no_dnn_dense_embedding_37.pth  --not_use_dnn --dense_embedding

python main.py --save_path ckpt/no_cin_dnn.pth --not_use_dnn --not_use_cin
