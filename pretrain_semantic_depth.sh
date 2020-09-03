SRC_ROOT=$1
TGT_ROOT=$2
python train.py --pretrain_semantic_module --n_train_epochs 10 --tgt_root $TGT_ROOT --src_root $SRC_ROOT --experiment_name 'pretrain' --loadSize 192 640 --batchSize 4