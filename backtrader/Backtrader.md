# Backtrader

backtrader是用以回测和交易的一个库

## 1. 主体架构

backtrader里的数据结构为lines（折线），即多个点构成的线。例如传入的数据为date,open,close,high,low,volume，backtrader把数据绘制成lines，通过对齐时间线进行执行策略。

```python
import backtrader as bt
cerebro = bt.Cerebro()
cerebro.adddata()
cerebro.addstrategy()
cerebro.run()
```

当达到交易条件时，**买入价格是下个bar的open**

## 2. 订单状态

order通过 self.buy() 方法获得，返回的order具有status属性，

order还有isbuy()，issell()方法，判断是买是卖。
$$
\begin{array}{l|lc}
order.status & 返回订单状态\\
order.Submitted & 已提交但未被 broker 接受 \\
order.Accepted & 被broker接受但尚未成交\\
order.Partial & 部分成交\\
order.Completed & 全部成交\\
order.Canceled & 被取消\\
order.Margin & 因保证金不足被取消\\
order.Rejected & 被broker拒绝\\
\end{array}
$$

$$
\begin{array}{l|lcc}
order.data & return对应的股票数据线\\
order.executed.size & return 成交数量\\
order.executed.price & return 执行价格\\
order.executed.value & return 执行金额\\
order.executed.comm & return 执行手续费\\
order.ref & return订单编号\\
order.getstatusname()&return状态名称（字符串）
\end{array}
$$

## 3. 数据导入

另外，**传入的数据要以date为索引**。通过pandas传入数据通过

```python
bt.feeds.PandasData(dataname=df,name='')
```

若是自定义了指标，不仅仅是用OHLCV来计算的指标，若是Pandas类型，可以通过重构PandasData

```python
class MyPandasData(bt.feeds.PandasData):
    lines = ('rsi1',) #填写在strategy中调用名
    params=(('rsi1','rsi'),)#填写这个名对应的是pandas的哪一列
```

同样，也可以通过继承bt.Indicator来计算，注意：**close(-self.p.period)**，用的是()来访问前值

```python
import backtrader as bt

class MyIndicator(bt.Indicator):
    lines = ('myline',)         # 声明输出线名称
    params = dict(period=20)

    def __init__(self):
        # 比如输出数据与收盘价差值
        self.l.myline = self.data.close - self.data.close(-self.p.period)
#策略中使用
self.myind = MyIndicator(self.data)
print(self.myind.myline[0])  # 当前值

```

`lines` 属性声明你要输出的线的名称，即使只有一条，也须用元组 `('myline',)` 。

在 `__init__` 中把计算结果赋给 `self.l.myline`（或 `self.lines.myline`）。

## 4. 运行机制

backtrader是按照bar来推进数据的，时间戳只是用来标识bar时间，每个bar有时间属性，self.datetime.datetime(0)。

当传入多个数据源时，cerebro自动同步多个时间，只处理当前时间最小的数据线，所以，某只股票缺失某一天的数据时，不会影响其回测。

Backtrader 会在 **多个数据源时间对齐后再执行 `next()`**。

- 如果某条数据在当前时间点**没有 bar（停牌）**，那一轮的 `next()` 不会触发；
- 会先执行 `prenext()` 阶段，直到所有数据源都同步到同一时间；
- **之后才进入 `next()`，且 `next()` 只在所有数据都有数据时才执行。**

运行机制

\__init\_\_ >>  start >> prenext >> nextstart >> next >> stop

假设你传入了一组日线数据（2024-01-01 ~ 2024-01-10），你会发现：

- `__init__()`：加载策略就会执行；
- `start()`：Cerebro 初始化 broker、cash、数据开始前执行；
- **`prenext()`**：如果多个数据源未完全对齐（或数据长度不同），先调用这个；
- **`nextstart()`**：一旦所有数据都对齐并且准备就绪，调用一次；
- **`next()`**：然后正式从这开始 bar-by-bar 推进；
- `stop()`：所有数据结束之后执行一次。

notify_orde()

在每个`next()`执行之前调用，

```scss
数据推进一根 bar →
→ 执行 notify_order()（若有订单状态变更）  
→ 执行 notify_trade()（若有交易成交）
→ 执行 next()（主策略逻辑）

当前 bar 到来：
 ├── check order updates → notify_order()
 ├── check trades       → notify_trade()
 └── 执行策略逻辑       → next()
```



```python
notify_order(self,order)bt.order.Order
notify_trade(self,trade)bt.trade.Trade
```

当订单发生变化时，或者每次成交时，backtrader会自动创建这两者对象，并作为参数返回到这两个函数中，这是回调机制的隐式传参。

```python
trade.status           # Trade 状态码
trade.isclosed         # 是否已平仓
trade.pnl              # 盈亏金额
trade.baropen          # 开仓时的 bar 索引
trade.barclose         # 平仓时的 bar 索引
trade.size             # 持仓股数（可能为负）
trade.price            # 平均成交价
```



