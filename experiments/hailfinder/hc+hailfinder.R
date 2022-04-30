
for (n in c(50,500,2000,5000))
{
  print(getwd())
  alarm <- read.csv("./experiments/data/Hailfinder/hailfinder.csv")
  alarm[,] <- lapply(alarm[,], as.factor)
  alarm = alarm[1:n,]

  # 结构学习
  dag = hc(alarm,score="bic",restart = 5000)

  # 转化为 graphNEL
  am = as.graphAM(dag)

  # 结果处理
  adjm = attr(am,'adjMat')
  row.names(adjm) <- colnames(adjm)
  write.csv (adjm,file = sprintf("./experiments/hailfinder/result/hc+hailfinder%d.csv",n))
}


