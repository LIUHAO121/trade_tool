export PYTHONPATH=${PYTHONPATH}:.
CUDA_VISIBLE_DEVICES=1


ts_code=002415.SZ
end_date=20220430

echo "back test $ts_code at date $end_date"

# python core/run_backtest.py --cfg experiments/reg_config_close_ma.json --ts_code $ts_code --end_date $end_date
python core/run_backtest.py --cfg experiments/reg_config_close_ma_tr.json --ts_code $ts_code --end_date $end_date