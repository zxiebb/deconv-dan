## Load in data
DIR<-"methylation/"
library(MethylResolver)
library(EMeth)
library(ICeDT)
library(FARDEEP)
library(ADAPTS)
dataset<-0
for (dataset in 0:19){
props_path<-paste(DIR,"target_EPIC_simulated/EPIC_simulated_test_y_",sep="")
props_path<-paste(props_path,dataset,sep="")
props_path<-paste(props_path,".csv",sep="")
props<-read.csv(props_path, row.names = 1)
ref_path<-paste(DIR,"target_EPIC_simulated/reference_EPIC_mean_value.csv",sep="")
ref <- read.csv(ref_path, row.names = 1)
target_path<-paste(DIR,"target_EPIC_simulated/EPIC_simulated_test_x_",sep="")
target_path<-paste(target_path,dataset,sep="")
target_path<-paste(target_path,".csv",sep="")
df <- read.csv(target_path, row.names = 1)
ref<-t(ref)
df<-t(df)
ref <- ref[rownames(df),]
  #EMeth-Normal-----
  deconv = 'emeth_normal'
  maximum_nu = 0
  maximum_iter = 50
  predicted_cell_proportions <- matrix(NA, ncol = dim(props)[2], nrow = dim(props)[1])
  predicted_cell_proportions <- emeth(Y = as.matrix(df[rownames(ref),]),eta = c(rep(0, dim(as.matrix(df[rownames(ref),]))[2])), mu = as.matrix(ref), aber = FALSE, V = 'c',init = "default", family = "normal", nu = maximum_nu, maxiter = maximum_iter, verbose = TRUE)$rho
  colnames(predicted_cell_proportions) <- colnames(props)
  rownames(predicted_cell_proportions) <- rownames(props)
  save_path<-paste(DIR,deconv,sep="")
  save_path<-paste(save_path,dataset,sep="")
  save_path<-paste(save_path,".csv",sep="")
  write.csv(predicted_cell_proportions,save_path)
  #EMeth-Laplace-----
  deconv = 'emeth_laplace'
  predicted_cell_proportions <- matrix(NA, ncol = dim(props)[2], nrow = dim(props)[1])
  predicted_cell_proportions <- emeth(Y = as.matrix(df[rownames(ref),]),eta = c(rep(0, dim(as.matrix(df[rownames(ref),]))[2])), mu = as.matrix(ref), aber = FALSE, V = 'c',init = "default", family = "laplace", nu = maximum_nu, maxiter = maximum_iter, verbose = TRUE)$rho
  colnames(predicted_cell_proportions) <- colnames(props)
  rownames(predicted_cell_proportions) <- rownames(props)
  save_path<-paste(DIR,deconv,sep="")
  save_path<-paste(save_path,dataset,sep="")
  save_path<-paste(save_path,".csv",sep="")
  write.csv(predicted_cell_proportions,save_path)
  #EMeth-Binom-----
  deconv = 'emeth_binom'
  predicted_cell_proportions <- matrix(NA, ncol = dim(props)[2], nrow = dim(props)[1])
  predicted_cell_proportions <- emeth(Y = as.matrix(df[rownames(ref),]),eta = c(rep(0, dim(as.matrix(df[rownames(ref),]))[2])), mu = as.matrix(ref), aber = FALSE, V = 'b',init = "default", family = "normal", nu = maximum_nu, maxiter = maximum_iter, verbose = TRUE)$rho
  colnames(predicted_cell_proportions) <- colnames(props)
  rownames(predicted_cell_proportions) <- rownames(props)
  save_path<-paste(DIR,deconv,sep="")
  save_path<-paste(save_path,dataset,sep="")
  save_path<-paste(save_path,".csv",sep="")
  write.csv(predicted_cell_proportions,save_path)
  #MethylResolver------
  deconv = 'methylresolver'
  methylMix <- as.data.frame(df[rownames(ref),])
  methylSig <- as.data.frame(ref)
  regressionFormula = as.formula(paste0("methylMix[,i] ~ ",paste(colnames(ref),sep="",collapse=" + ")))
  preds <- matrix(NA, ncol = length(colnames(ref)), nrow = length(colnames(df)))
  colnames(preds) <- colnames(ref)
  rownames(preds) <- colnames(df)
  j = 1
  for (i in colnames(df)){
    deconvoluteSample <- robustbase::ltsReg(regressionFormula, data = methylSig, alpha = 0.5)
    preds[j,] <- deconvoluteSample$coefficients[2:length(deconvoluteSample$coefficients)]
    j <- j+1
  }
  predicted_cell_proportions <- preds
  save_path<-paste(DIR,deconv,sep="")
  save_path<-paste(save_path,dataset,sep="")
  save_path<-paste(save_path,".csv",sep="")
  write.csv(predicted_cell_proportions,save_path)
  #ICeDT------
  deconv = 'icedt'
  predicted_cell_proportions <- as.data.frame(t(ICeDT(as.matrix(df[rownames(ref),]), as.matrix(ref), rhoConverge = 0.0015, maxIter_PP = 30)$rho))[,colnames(props)]
  save_path<-paste(DIR,deconv,sep="")
  save_path<-paste(save_path,dataset,sep="")
  save_path<-paste(save_path,".csv",sep="")
  write.csv(predicted_cell_proportions,save_path)
  #FARDEEP------
  deconv = 'fardeep'
  predicted_cell_proportions <- as.data.frame(fardeep(as.matrix(ref),as.matrix(df[rownames(ref),]))$abs.beta)[,colnames(props)]
  save_path<-paste(DIR,deconv,sep="")
  save_path<-paste(save_path,dataset,sep="")
  save_path<-paste(save_path,".csv",sep="")
  write.csv(predicted_cell_proportions,save_path)
  #DCQ------
  deconv = 'dcq'
  predicted_cell_proportions <- as.data.frame(t(estCellPercent.DCQ(refExpr=as.matrix(ref), geneExpr=as.matrix(df[rownames(ref),]))))[,colnames(props)]/100
  save_path<-paste(DIR,deconv,sep="")
  save_path<-paste(save_path,dataset,sep="")
  save_path<-paste(save_path,".csv",sep="")
  write.csv(predicted_cell_proportions,save_path)
  
}

  