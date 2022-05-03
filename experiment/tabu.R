asia <- read.csv("bnsl/datasets/asian/asian.csv")
asia<-asia [,-1]
asia[,] <- lapply(asia[,], as.factor)
alarm <- read.csv("bnsl/datasets/alarm/alarm.csv")
alarm <- alarm[,-1]
alarm[,] <- lapply(alarm[,], as.factor)
insurance <- read.csv("bnsl/datasets/insurance/insurance.csv")
insurance <- insurance[,-1]
insurance[,] <- lapply(insurance[,], as.factor)
hailfinder <- read.csv("bnsl/datasets/hailfinder/hailfinder.csv")
hailfinder <- hailfinder[,-1]
hailfinder[,] <- lapply(hailfinder[,], as.factor)

for (n in c(50,200,2000,5000)){
  asia_data <- asia[1:n,]
  alarm_data <- alarm[1:n,]
  insurance_data <- insurance[1:n,]
  hailfinder_data <- hailfinder[1:n,]
  asia_result <- tabu(asia_data)
  asia_g <- as.graphAM(asia_result)
  adjm <- attr(asia_g,'adjMat')
  row.names(adjm) <- colnames(adjm)
  write.csv (adjm,file = sprintf("experiment/result/asia-%d-tabu.csv",n))
  # alarm
  alarm_result <- tabu(alarm_data)
  alarm_g <- as.graphAM(alarm_result)
  adjm <- attr(alarm_g,'adjMat')
  row.names(adjm) <- colnames(adjm)
  write.csv (adjm,file = sprintf("experiment/result/alarm-%d-tabu.csv",n))
  # insurance
  insurance_result <- tabu(insurance_data)
  insurance_g <- as.graphAM(insurance_result)
  adjm <- attr(insurance_g,'adjMat')
  row.names(adjm) <- colnames(adjm)
  write.csv (adjm,file = sprintf("experiment/result/insurance-%d-tabu.csv",n))

  # hailfinder
  hailfinder_result <- tabu(hailfinder_data)
  hailfinder_g <- as.graphAM(hailfinder_result)
  adjm <- attr(hailfinder_g,'adjMat')
  row.names(adjm) <- colnames(adjm)
  write.csv (adjm,file = sprintf("experiment/result/hailfinder-%d-tabu.csv",n))

}
