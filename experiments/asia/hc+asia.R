# 爬山算法+asia

for (n in c(50,500,2000,5000))
{
  print(getwd())
  asia <- read.csv("experiments/data/Asia/Asian.csv")
  asia[,] <- lapply(asia[,], as.factor)
  asia <- asia[1:n,]

  # 结构学习
  dag <- hc(asia, score="bic", restart = 5000)

  # 转化为 graphNEL
  am <- as.graphAM(dag)

  # 结果处理
  adjm <- attr(am, 'adjMat')
  row.names(adjm) <- colnames(adjm)
  write.csv (adjm,file = sprintf("experiments/asia/result/hc+asia%d.csv",n))
}
