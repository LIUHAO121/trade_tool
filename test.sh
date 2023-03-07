export PYTHONPATH=${PYTHONPATH}:.
CUDA_VISIBLE_DEVICES=1 
ts_code=002415
python core/test.py --cfg config/reg_config_close_ma_tr.json  --ts_code $ts_code