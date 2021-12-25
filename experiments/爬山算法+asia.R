# 数据读取
asia <- read.csv("./experiments/data/Asia/Asian.csv")
asia[,] <- lapply(asia[,], as.factor)
asia = asia[1:50,]

# 结构学习
dag = hc(asia,score="bic",restart = 0)

# 结果处理
result = arcs(dag)
colnames(result) <- c("source node","target node")
write.csv (result,file = "./experiments/result/hc+asia50.csv")
