
for (n in c(50,500,2000,5000))
{
  print(getwd())
  alarm <- read.csv("./experiments/data/Alarm/alarm.csv", row.names = 1)
  alarm[,] <- lapply(alarm[,], as.factor)
  alarm = alarm[1:n,]

  # 结构学习
  dag = tabu(alarm,score="bic")

  # 转化为 graphNEL
  am = as.graphAM(dag)

  # 结果处理
  adjm = attr(am,'adjMat')
  row.names(adjm) <- colnames(adjm)
  write.csv (adjm,file = sprintf("./experiments/alarm/result/tabu+alarm%d.csv",n))
}