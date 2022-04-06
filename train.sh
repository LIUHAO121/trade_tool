export PYTHONPATH=${PYTHONPATH}:.
CUDA_VISIBLE_DEVICES=1 
python core/train.py --cfg experiments/reg_config_close.json
# python core/train.py --cfg experiments/reg_config_open_high_low_close.json
