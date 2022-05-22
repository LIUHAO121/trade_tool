import matplotlib.pyplot as plt

# operate
KEEP = 0
BUY = 1
SELL = 2


op2string = {
    KEEP:"keep",
    BUY:"buy",
    SELL:"sell",
}

class BackTest(object):
    def __init__(self,config,model,dataset):
        """
        目前只支持回测一只股票
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.market_value = 0   # 市场价值，当市场价值为0时说明没有持有股票
        self.keep_number = 0  # 持有股票数量
        self.wallet = self.config["backtest"]["wallet"]  # 起始资金
        self.min_buy_number = self.config["backtest"]["min_buy_number"] # 最少购买股票数量
        self.min_keep_days = self.config["backtest"]["min_keep_days"]  # 买入后最少持有时间，当该值为1时，代表今天买的股票最早能明天卖
        # draw
        self.seq_len = self.dataset.seq_len
        self.buy_points = [None for i in range(self.seq_len)]
        self.sell_points = [None for i in range(self.seq_len)]
        
    
    def run(self,log):
        rows,_ = self.dataset.stock_df.shape
        seq_len = self.dataset.seq_len
        assert rows > seq_len
        # judge_times = (rows - seq_len) + 1  
        judge_times = rows - seq_len # 预测完最后一天就停止
        for i in range(judge_times):
            input_sample = self.dataset.stock_df.iloc[i:i+seq_len]
            pred_res = self.dataset.predict_one_sample(self.model, input_sample)
            op_signal = self.opration_signal(pred_res)
            self.op_signal = op_signal
            self.after_signal(op_signal, timestamp=i+seq_len) # 用前N天的数据, 为第N+1天做判断
            log_info = "timestamp:{}, op:{}, wallet:{:.2f}, market_value:{:.2f}, keep_num:{}, close:{:.2f}".format(i+seq_len,op2string[self.op_signal],self.wallet,self.market_value,self.keep_number,self.current_close_price)
            log.info(log_info)
            self.collect_draw_point(draw=False)
        self.collect_draw_point(draw=True)
        
    def collect_draw_point(self,draw):
        if draw:
            close_prices = list(self.dataset.original_stock_df.loc[:,'close'].values) 
            
            fig = plt.figure(facecolor='white')

            ax = fig.add_subplot(111)
            ax.plot(close_prices, label='close')
            ax.plot(self.buy_points,label='buy')
            ax.plot(self.sell_points,label='sell')
            plt.legend()
            plt.savefig("log/{}.png".format("backtest"))

        if self.op_signal == BUY:
            self.buy_points.append(self.current_close_price * 1.1)
            self.sell_points.append(None)
        elif self.op_signal == SELL:
            self.sell_points.append(self.current_close_price * 0.9)
            self.buy_points.append(None)
        else:
            self.sell_points.append(None)
            self.buy_points.append(None)
            
            
    def after_signal(self,signal,timestamp):
        """
        Args:
            signal (int): buy ,sell
            timestamp (int): 进行交易的时间
            
        如果交易信号为买,那就买入最少数量的股票
        如果交易信号为卖,那就将所有股票卖出
        """
        current_open_price = self.dataset.original_stock_df.loc[timestamp,'open']
        current_close_price = self.dataset.original_stock_df.loc[timestamp,'close']   
        
        if signal == BUY:
            need_money = current_open_price * self.min_buy_number
            if self.wallet >= need_money:
                self.wallet -= need_money
                self.keep_number += self.min_buy_number
                
        elif signal == SELL:
            sell_money = self.keep_number * current_open_price
            self.wallet += sell_money
            self.keep_number -= self.keep_number
            
        # update market value  
        self.market_value = current_close_price * self.keep_number
        self.current_close_price = current_close_price
        
    def opration_signal(self,one_pred_res):
        sample_len = len(one_pred_res)
        index = int(sample_len/2)
        if min(one_pred_res[:index]) >= one_pred_res[0]:
            return BUY
        else:
            return SELL
    
  
    def run_double_ma(self,log,short_average="ma5",long_average="ma15"):
        rows,_ = self.dataset.stock_df.shape
        seq_len = 2
        judge_times = rows - seq_len # 预测完最后一天就停止
        for i in range(judge_times):
            op_signal=None
            short_1 = self.dataset.original_stock_df.loc[i,short_average]
            long_1 = self.dataset.original_stock_df.loc[i,long_average]
            short_2 = self.dataset.original_stock_df.loc[i+1,short_average]
            long_2 = self.dataset.original_stock_df.loc[i+1,long_average]
            if short_1 < long_1 and short_2 > long_2:
                op_signal = BUY
            elif short_1 > long_1 and short_2 < long_2:
                op_signal = SELL
            else:
                op_signal = KEEP
            
            self.op_signal = op_signal
            self.after_signal(op_signal, timestamp=i+seq_len) # 用前N天的数据，为第N+1天做判断
            log_info = "timestamp:{}, op:{}, wallet:{:.2f}, market_value:{:.2f}, keep_num:{}, close:{:.2f}".format(i+seq_len,op2string[self.op_signal],self.wallet,self.market_value,self.keep_number,self.current_close_price)
            log.info(log_info) 
