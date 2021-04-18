library(tseries)
library(forecast)
# 设置工作目录并读取数据
setwd('H:\\Program Products\\Python Files\\0 Jupyter\\FDM-Project-2021\\dataset\\期货价格数据\\使用数据\\')

da <- read.csv("./黄金周数据.csv",head=F)
names(da) <- c('date','price')
head(da)

price <- da$price
price = ts(da$price,frequency=52,start=c(1972,1,10))
head(price)
plot.ts(price)

tsdisplay(price)

# 拆掉最后一年做样本的测试集
sprice<-ts(as.vector(price[1:500]),frequency=52,start=c(1972,1,10))
tsdisplay(sprice)
adf.test(sprice)

# 明显存在按经济波动增长的趋势，故利用差法将其干掉
s1<-diff(sprice,1)
# 单位根检验判断是否平稳
adf.test(s1)
# 检验通过
tsdisplay(s1)

# 图像显示acf与pac均存在截尾情况
# 根据pacf图像可判断 p 取 2 或 100
# 根据acf图像可判断 q 取 2 90 或 140

# 进行模型拟合，看看哪个效果好
a <- auto.arima(sprice)
summary(a)

arima(sprice,order=c(2,1,2))
arima(sprice,order=c(100,1,2))
arima(sprice,order=c(2,1,90))
arima(sprice,order=c(2,1,140))
arima(sprice,order=c(100,1,120))
arima(sprice,order=c(100,1,90))

#先进行拟合
fit1<-arima(sprice,order=c(2,1,2),seasonal=list(order=c(1,0,0),period=52))
#然后tsdiag看一下各自的结果,Ljung-Box检验的p值都在0.05之上，结果不错。
tsdiag(fit1)
fit1

#预测
f.p1<-forecast(fit1,h=50,level=c(99.5))
plot(f.p1,ylim=c(100,500))
lines(f.p1$fitted,col="green")
lines(price,col="red")

# 利用全部数据进行建模

#先进行拟合
fit2<-arima(price,order=c(2,1,2),seasonal=list(order=c(1,0,0),period=52))
tsdiag(fit2)

#预测
f.p2<-forecast(fit2,h=7,level=c(99.5))
plot(f.p2,ylim=c(100,500))
lines(f.p2$fitted,col="green")
lines(price,col="red")
f.p2