# 数据读取
print(getwd())
asia <- read.csv("datasets/asian/Asian.csv")
asia[,] <- lapply(asia[,], as.factor)
asia = asia[1:50,]

# 结构学习
dag = hc(asia,score="bic",restart = 0)

# 转化为 graphNEL
am = as.graphAM(dag)

# 结果处理
adjm = attr(am,'adjMat')
row.names(adjm) <- colnames(adjm)
write.csv (adjm,file = "./experiments/result/hc+asia50.csv")
