# 设置工作目录并读取数据
setwd('H:\\Program Products\\Python Files\\0 Jupyter\\FDM-Project-2021\\dataset\\CarPrice')

da <- read.csv("./CarPrice-n.csv",head=T)
# names(da) <- c('date','price')
da.all <- da
da <- da[-1]
head(da.all)

pairs(as.data.frame(da))

# 相关系数矩阵
library(corrplot)
library(RColorBrewer)
M <- cor(da)
corrplot(M,type = "upper", # 只显示上三角
         order = "hclust",col = brewer.pal(n = 8,name = "RdYlBu"))

# 建立多元线性回归模型
car.lm <- lm(price~.,data=da)

car.lm

summary(car.lm)

par(mfrow=c(2,2))
plot(car.lm)



# 可以看到明显的异常点17，50，129，130
da.new <- da[c(-17,-50,-129,-130),]
head(da.new)

# 建立多元线性回归模型
car.lm.new <- lm(price~.,data=da.new)

car.lm.new

summary(car.lm.new)

par(mfrow=c(2,2))
plot(car.lm.new)

# 逐步回归
car.lm.new.step <- step(car.lm.new,direction="both")

car.lm.new.step

summary(car.lm.new.step)

par(mfrow=c(2,2))
plot(car.lm.new.step)

# 异方差检验与修正
par(mfrow=c(1,1))
e<-resid(car.lm.new.step)  # 计算残差
attach(da.all)
plot(car_ID[c(-17,-50,-129,-130)],e)
abline(h=c(0),lty=5)  # 添加虚线e=0
detach(da.all)

abse<-abs(e)
cor.test(da.all$car_ID[c(-17,-50,-129,-130)],abse,alternative="two.sided",method="spearman",conf.level=0.95)

# 异方差修正 加权最小二乘
fit <- lm(log(resid(car.lm.new.step)^2) ~ wheelbase + carlength + carwidth + carheight + curbweight + 
              enginesize + boreratio + stroke + compressionratio + horsepower + 
              peakrpm + citympg + highwaympg, data=da.new)
fit2 = lm(price ~ wheelbase + carlength + carwidth + carheight + curbweight + 
              enginesize + boreratio + stroke + compressionratio + horsepower + 
              peakrpm + citympg + highwaympg, data=da.new,weights=(1/exp(fitted(fit))))
fit2
summary(fit2)

par(mfrow=c(2,2))
plot(fit2)

library('GGally')
ggpairs(data=da.new)

ans <- predict(fit2,interval="prediction",level=0.95)

plot(ans)

plot(da.all$car_ID[c(-17,-50,-129,-130)],da.all$price[c(-17,-50,-129,-130)])
lines(da.all$car_ID[c(-17,-50,-129,-130)],ans[,1],lty=1,col="red")
lines(da.all$car_ID[c(-17,-50,-129,-130)],ans[,2],lty=3,col="blue")
lines(da.all$car_ID[c(-17,-50,-129,-130)],ans[,3],lty=3,col="blue")