CUDA_VISIBLE_DEVICES=$1 python train_gazegaussian.py \
--batch_size 1 \
--name 'gazegaussian' \
--img_dir './data/ETH-XGaze' \
--num_epochs 20 \
--num_workers 2 \
--lr 0.0001 \
--clip_grad \
--load_gazegaussian_checkpoint ./checkpoint/gazegaussian_ckp.pth \
# --load_meshhead_checkpoint ./work_dirs/meshhead/checkpoints/meshhead_epoch_10.pth \
