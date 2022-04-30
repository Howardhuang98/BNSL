# mmhc+insurance

for (n in c(50,500,2000,5000))
{
  # 读取数据
  print(getwd())
  asia <- read.csv("experiments/data/Insurance/insurance.csv")
  asia[,] <- lapply(asia[,], as.factor)
  asia <- asia[1:n,]

  # 结构学习
  dag <- mmhc(asia)

  # 转化为 graphNEL
  am <- as.graphAM(dag)

  # 结果处理
  adjm <- attr(am, 'adjMat')
  row.names(adjm) <- colnames(adjm)
  write.csv (adjm,file = sprintf("experiments/insurance/result/mmhc+insurance%d.csv",n))
}