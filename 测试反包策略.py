import backtrader as bt
from datetime import datetime
import tushare as ts
import pandas as pd


#通过tushare获取数据
def datas_from_tusharepro(code, start_date, end_date):
    pro = ts.pro_api("068317422c70e7e12e70b80d2d0ed42837c0b970e554e2e3e8a76a69")
    df = pd.DataFrame([[]])
    #构建股票代码
    if code[:2] == '60':
        code_ = code + '.SH'
    elif code[:2] == '00':
        code_ = code + '.SZ'
    else:
        print("不采集科创板的股票")
        return df
    df = pro.daily(ts_code=code_, start_date=start_date, end_date=end_date)
    df.index = pd.to_datetime(df.trade_date)
    df = df[::-1]
    df['openinterest'] = 0
    df['volume'] = df.vol
    return df[['open', 'high', 'low', 'close', 'volume', 'openinterest']]


success = 0
total_ = 0


class Limit(bt.Indicator):
    lines = ('limit',)
    """
    ZT:=C>1.1*REF(C,1)-0.01 AND C<1.1*REF(C,1)+0.01 AND H=C;
    """

    def __init__(self):
        close = self.data.close
        high = self.data.high
        condition1 = close == high
        condition2 = close > 1.1 * close(-1) - 0.01
        condition3 = close < 1.1 * close(-1) + 0.01
        self.lines.limit = bt.And(condition1, condition2, condition3)
        super(Limit, self).__init__()


class MyStrategy(bt.Strategy):
    params = dict(
        #持仓够5个单位就卖出
        exitbars=1,
        printlog=False
    )

    #打印函数
    def log(self, txt, dt=None, doprint=False):
        # 记录策略的执行日志
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.close = self.data.close  #datas[0]的意思是访问第一个data，如果加入数据源有两个就data[1]
        self.open = self.data.open
        self.high = self.data.high
        #order订单详情
        self.order = None

        self.limit = Limit(self.data)
        #order订单详情
        self.order = None
        # 买入价格和手续费
        self.buyprice = None
        self.buycomm = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # broker 提交/接受了，买/卖订单则什么都不做
            return
            # 检查一个订单是否完成
            # 注意: 当资金不足时，broker会拒绝订单
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    '已买入, 价格: %.2f, 费用: %.2f, 佣金 %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
            #     self.buyprice = order.executed.price
            #     self.buycomm = order.executed.comm
            elif order.issell():
                #     self.log('已卖出, %.2f' % order.executed.price)
                self.log('已卖出, 价格: %.2f, 费用: %.2f, 佣金 %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
            # 记录当前交易数量
            self.bar_executed = len(self)

        # 记录当前交易数量

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')

            # 其他状态记录为：无挂起订单
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('交易利润, 毛利润 %.2f, 净利润 %.2f' %
                 (trade.pnl, trade.pnlcomm))  #毛利率和净利润

    def next(self):
        # self.log('Close, %.2f' % self.close[0])
        # 如果有订单正在挂起，不操作
        if self.order:
            return

        if not self.position:  #没有持仓则买入
            if self.limit[0] == 1.0 and self.limit[-1] == 0.0 and self.limit[-2] == 1.0:
                self.log('买入, %.2f' % self.close[0])  # 买入
                #跟踪订单，防止重复
                self.order = self.buy()  #买入价格是下一根K线的开盘价

        else:
            # 如果已经持仓，且当前交易数据量在买入后5个单位后
            """
            没有将柱的下标传给 next()方法，怎么知道已经经过了5个柱了呢？ 这里用了 Python的len()方法获取它Line数据的长度。
             交易发生时记下它的长度，后边比较大小，看是否经过了5个柱。
            """
            if len(self) >= (self.bar_executed + self.params.exitbars):
                # 全部卖出
                self.log('卖出, %.2f' % self.close[0])
                global success, total_
                total_ += 1
                if (self.close[0] - self.close[-self.params.exitbars]) / self.close[-self.params.exitbars] > 0.03:
                    success += 1
                # 跟踪订单避免重复
                self.order = self.sell()

    def stop(self):
        self.log('(均线周期 %2d)期末资金 %.2f' %
                 (self.params.exitbars, self.broker.getvalue()), doprint=False)


stock_count = 0
stock_code = pd.read_csv(r"主板.csv", dtype={'代码': 'str'})
stocks = stock_code['代码']
for stock in stocks:
    try:
        #制造数据
        stock_data = datas_from_tusharepro(stock, '20190101', '20220101')
        start_time = datetime(2019, 1, 1)
        end_time = datetime(2022, 1, 1)
        data = bt.feeds.PandasData(dataname=stock_data, fromdate=start_time, todate=end_time)
        #创建大脑
        cerebro = bt.Cerebro()
        # 为Cerebro引擎添加策略, 优化策略
        # 使用参数来设定10到31天的均线,看看均线参数下那个收益最好
        # strats = cerebro.optstrategy(MyStrategy, smaperiod=20, exitbars=range(3, 30), printlog=False)  #range(10, 31)
        #加入数据
        cerebro.adddata(data)
        #加入策略
        cerebro.addstrategy(MyStrategy)
        #设置经纪人，初始现金
        cerebro.broker.setcash(100000)
        # 每笔交易使用固定交易量
        cerebro.addsizer(bt.sizers.FixedSize, stake=10)
        #设置交易费用
        cerebro.broker.setcommission(commission=0.001)
        #打印初始资金
        # print("初始资金: %.2f" % cerebro.broker.getvalue())
        #执行

        cerebro.run(maxcpus=1)  #
        # print('期末资金: %.2f' % cerebro.broker.getvalue())
        # cerebro.plot()
        stock_count += 1
        if stock_count % 100 == 0:
            print(f'正在回测第{stock_count}支股票')
    except:
        print(f'第{stock_count}支股票出了错')
print(f"信号出现次数{total_},成功次数{success}")
print("成功率：%.2f" % (100 * success / total_))
