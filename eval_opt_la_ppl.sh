device=$1
pretrain_dir=$2
T=$3

CUDA_VISIBLE_DEVICES=$device python read_inference_model_lt.py --config_path configs_345m_single_card_chek.yaml --pretrain_dir $pretrain_dir --T $T
