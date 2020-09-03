SRC_ROOT=$1
TGT_ROOT=$2
python train.py --train_image_generator --tgt_root $TGT_ROOT --src_root $SRC_ROOT --experiment_name 'pretrain' --loadSize 192 640