

#twogroupBANOVA <- function(analyzed_data){

  library(BayesFactor)
  library(Matrix)
  library(reshape)
  library(coda)
 
  path="/PARA2/paratera_blsca_056/wbl/" 
  setwd(path)
  #argv <- commandArgs(TRUE) /PARA2/paratera_blsca_056/wbl/SUMO_RPA_2dim
  #value=argv[1]
  #filename<-paste("analyzed_data",value,".csv", sep="")
  filename<-"analyzed_data.csv"
  analyzed_data <- read.table(filename, header = FALSE, sep=",")
  data_0<-analyzed_data  
  
  
  data_0$V1 <- factor(data_0$V1)

  mod.eq <- BayesFactor::lmBF(V2 ~ V1        
                              , data = data_0)
  est.eq <- BayesFactor::posterior(mod.eq, iterations = 20000)
  
  priorPeq <- 1/2 
  
  is.greater <- function(mu){
    mu[1] > mu[2]
  }
  
  only_spec_inter <- est.eq[, "V1-1"]
  except_spec_inter <- est.eq[, "V1-0"]
  
  
  res <- apply(cbind(only_spec_inter, except_spec_inter), 1, is.greater)
  postPeq <- mean(res)
  
  
  BF_spec_inter <- postPeq / priorPeq

  BF_ru <-  BF_spec_inter
  
  BF_ru 

  #filename2<-paste("BF_ru",value,".csv", sep="")
  filename2<-"BF_ru.csv"
  write.table (BF_ru,  filename2,row.names=FALSE,col.names=FALSE,sep=",")

