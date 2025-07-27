CUDA_VISIBLE_DEVICES=$1 python train_meshhead.py \
--batch_size 1 \
--name 'meshhead' \
--img_dir './data/ETH-XGaze' \
--num_epochs 10 \
--num_workers 8 \
