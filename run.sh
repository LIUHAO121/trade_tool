export PYTHONPATH=${PYTHONPATH}:.
python core/train.py --cfg experiments/config.json
# python core/train.py --cfg experiments/config_less_columns.json