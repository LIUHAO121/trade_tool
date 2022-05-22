export PYTHONPATH=${PYTHONPATH}:.
CUDA_VISIBLE_DEVICES=1 

# data dir
train_dir=data/train_qfq
test_dir=data/test_qfq
src_dir=data/qfq

# clean up old data
rm -rf ${train_dir}/*
rm -rf ${test_dir}/*

# need data
# train_code=(000004 000158 000409 000503 000555 000938 000948 000997 002230 002415 )
train_code=(002415)
test_code=002415


# copy need data
echo "cp train data ..."
for i in ${train_code[*]};
do 
    cp ${src_dir}/${i}* $train_dir
    echo "cp ${i} to $train_dir"
done

echo "cp test data ..."
cp ${src_dir}/${test_code}* $test_dir
echo "cp ${test_code} to ${test_dir}"

# python core/train.py --cfg experiments/reg_config_close.json --ts_code $test_code
# python core/train.py --cfg experiments/reg_config_close_ma.json --ts_code $test_code
python core/train.py --cfg experiments/reg_config_close_ma_tr.json --ts_code $test_code