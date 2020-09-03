SRC_ROOT=$1
TGT_ROOT=$2
python train.py --load_pretrained --use_semantic_const --n_train_iterations 20000 --tgt_root $TGT_ROOT --src_root $SRC_ROOT --experiment_name 'joint_training' --loadSize 192 640