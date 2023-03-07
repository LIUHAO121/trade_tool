# trade_tool

这是一个利用深度学习预测股票未来价格的项目

### env

```
pip install -r requirements.txt
mkdir log weights
mkdir data/test_qfq data/test_train data/train_qfq
```

mkdir log weights


### train

训练某只股票的时序模型

```shell
bash train.sh
```

### test

可视化模型维度的表现

```
bash test.sh
```

### back test

将模型输出作为交易信号在历史数据上测试

```shell
bash backtest.sh
```
