export PYTHONPATH=${PYTHONPATH}:.
CUDA_VISIBLE_DEVICES=1 
python core/test.py --cfg experiments/reg_config_close.json