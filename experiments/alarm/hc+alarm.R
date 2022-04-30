
for (n in c(50,500,2000,5000))
{
  print(getwd())
  alarm <- read.csv("./experiments/data/Alarm/alarm.csv")
  alarm[,] <- lapply(alarm[,], as.factor)
  alarm = alarm[1:n,]

  # 结构学习
  dag = hc(alarm,score="bic",restart = 500)

  # 转化为 graphNEL
  am = as.graphAM(dag)

  # 结果处理
  adjm = attr(am,'adjMat')
  row.names(adjm) <- colnames(adjm)
  write.csv (adjm,file = sprintf("./experiments/alarm/result/hc+alarm%d.csv",n))
}


