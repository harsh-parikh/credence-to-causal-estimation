## 2021 selective ML simulations
library(randomForest)
library(gbm)
library(caret)
library(glmnet)
library(tmle)
library(ctmle)
library(boot)
library(SuperLearner)
f <- function(outcome, treatment, data){

# redefine SL.gbm to use one core
  SL.gbm.onecore = function(...) {
    SL.gbm(..., n.cores = 1)
  }

  y <- data[outcome]
  A <- data[treatment]
  x <- data[-c(outcome,treatment)]
  # f0 = function(a)
  # { 
  #   return(1/(1+exp(-20*(a-0.5))))
  # }

  #### true coefficient ####
  # beta0 <- 2*1
  # beta1 <- 2*c(1,1,1,1,1)
  # beta2 <- 2*c(1,1,1,1,1)
  # beta3 <- 2*1
  # alpha0 <- 0
  # alpha1 <- c(1,-1,1,-1,1)

  K <- 4  # K^2 candidates
  # m <- 5
  # n <- 250
  # sigma <- 1
  Q <- 100
  nsplit <- 2

  psi.save <-  array(NA, c(Q, K, K)) ### psi estimation for each q
  psi.select1 <- rep(NA,Q)
  psi.select2 <- rep(NA,Q)
  psi.lasso <- rep(NA,Q)
  psi.rf <- rep(NA,Q)
  psi.gbm <- rep(NA,Q)
  psi.sup <- rep(NA,Q)
  psi.tmle0 <- rep(NA,Q)
  psi.tmle1 <- rep(NA,Q)
  psi.tmle2 <- rep(NA,Q)
  select1 <- matrix(NA,Q,2)
  select2 <- matrix(NA,Q,2)

  cond.select1.lower <- rep(NA,Q)
  cond.select2.lower <- rep(NA,Q)
  psi.select1.lower <- rep(NA,Q)
  psi.select2.lower <- rep(NA,Q)
  psi.gbm.lower <- rep(NA,Q)
  psi.rf.lower <- rep(NA,Q)
  psi.lasso.lower <- rep(NA,Q)
  psi.sup.lower <- rep(NA,Q)
  psi.tmle0.lower <- rep(NA,Q)
  psi.tmle1.lower <- rep(NA,Q)
  psi.tmle2.lower <- rep(NA,Q)

  cond.select1.upper <- rep(NA,Q)
  cond.select2.upper <- rep(NA,Q)
  psi.select1.upper <- rep(NA,Q)
  psi.select2.upper <- rep(NA,Q)
  psi.gbm.upper <- rep(NA,Q)
  psi.rf.upper <- rep(NA,Q)
  psi.lasso.upper <- rep(NA,Q)
  psi.sup.upper <- rep(NA,Q)
  psi.tmle0.upper <- rep(NA,Q)
  psi.tmle1.upper <- rep(NA,Q)
  psi.tmle2.upper <- rep(NA,Q)

  for (q in 1:Q){
    set.seed(2021*q)
    ##### initial bias ####
    psi.est <- array(NA, c(nsplit, K, K))
    psi.avg <- array(NA, c(1,K,K))
    bias.ii <- array(NA, c(K, K))
    bias.jj <- array(NA, c(K, K))
    bias.i <- array(0, c(K, K, K))
    bias.j <- array(0, c(K, K, K))
    bias1 <- array(NA, c(1, K, K))
    bias2 <- array(NA, c(1, K, K))

  ######### Generate data #########
  # x <- matrix(runif(n*m),nrow=n,ncol=m)
  # p.A <- inv.logit(alpha0 + f0(x)%*%alpha1)
  # A <- rbinom(n,1,p.A)
  # mu <- beta0 + f0(x)%*%beta1 + (f0(x)*A)%*%beta2 + beta3*A
  # y <- mu + rnorm(n,0,sigma)

  ##########  fit model separately for 4 #######
  train_ind <- rbinom(n,1,0.5)

  x.train <- x[train_ind==0,]
  A.train <- A[train_ind==0]
  y.train <- y[train_ind==0]

  x.test <- x[train_ind==1, ]
  A.test <- A[train_ind==1]
  y.test <- y[train_ind==1]

  ######################## 1.lasso 2.rf 3.gbm 4.super learner ########################
  df <- data.frame(x, A)
  matrix.train <- data.frame(x.train, A=A.train)
  matrix.test <- data.frame(x.test, A=A.test)
  model <- SuperLearner(Y = y.train, X = matrix.train, SL.library = c("SL.glmnet","SL.randomForest","SL.gbm.onecore")) 
  model.pred <- predict(model, newdata=matrix.test)
  y.test1 <- model.pred$library.predict[,1]
  y.test2 <- model.pred$library.predict[,2]
  y.test3 <- model.pred$library.predict[,3]
  y.test4 <- model.pred$pred

  df1 <- data.frame(cbind(x.test, A=1))
  names(df1) <- names(df)
  model.pred1 <- predict(model, newdata=df1)
  y1.test1 <- model.pred1$library.predict[,1]
  y1.test2 <- model.pred1$library.predict[,2]
  y1.test3 <- model.pred1$library.predict[,3]
  y1.test4 <- model.pred1$pred

  df0 <- data.frame(cbind(x.test, A=0))
  names(df0) <- names(df)
  model.pred0 <- predict(model, newdata=df0)
  y0.test1 <- model.pred0$library.predict[,1]
  y0.test2 <- model.pred0$library.predict[,2]
  y0.test3 <- model.pred0$library.predict[,3]
  y0.test4 <- model.pred0$pred

  Amatrix.train <- data.frame(x.train)
  Amatrix.test <- data.frame(x.test)
  Amodel <- SuperLearner(Y = A.train, X = Amatrix.train, family = binomial(), SL.library = c("SL.glmnet","SL.randomForest","SL.gbm.onecore"))
  Amodel.pred <- predict(Amodel, newdata=Amatrix.test)
  A.test1 <- Amodel.pred$library.predict[,1]
  A.test2 <- Amodel.pred$library.predict[,2]
  A.test3 <- Amodel.pred$library.predict[,3]
  A.test4 <- Amodel.pred$pred

  temp1.1 <- mean( (y.test-y.test1)*(-1)^(1-A.test)/((A.test1)^A.test*(1-A.test1)^(1-A.test)) + y1.test1 - y0.test1)
  temp1.2 <- mean( (y.test-y.test2)*(-1)^(1-A.test)/((A.test2)^A.test*(1-A.test2)^(1-A.test)) + y1.test2 - y0.test2)
  temp1.3 <- mean( (y.test-y.test3)*(-1)^(1-A.test)/((A.test3)^A.test*(1-A.test3)^(1-A.test)) + y1.test3 - y0.test3)
  temp1.4 <- mean( (y.test-y.test4)*(-1)^(1-A.test)/((A.test4)^A.test*(1-A.test4)^(1-A.test)) + y1.test4 - y0.test4)

  tempvar1.1 <- var((y.test-y.test1)*(-1)^(1-A.test)/((A.test1)^A.test*(1-A.test1)^(1-A.test)) + y1.test1 - y0.test1 - temp1.1 )/length(y.test)
  tempvar1.2 <- var((y.test-y.test2)*(-1)^(1-A.test)/((A.test2)^A.test*(1-A.test2)^(1-A.test)) + y1.test2 - y0.test2 - temp1.2 )/length(y.test)
  tempvar1.3 <- var((y.test-y.test3)*(-1)^(1-A.test)/((A.test3)^A.test*(1-A.test3)^(1-A.test)) + y1.test3 - y0.test3 - temp1.3 )/length(y.test)
  tempvar1.4 <- var((y.test-y.test4)*(-1)^(1-A.test)/((A.test4)^A.test*(1-A.test4)^(1-A.test)) + y1.test4 - y0.test4 - temp1.4 )/length(y.test)


  ########################## swap #####################
  model <- SuperLearner(Y = y.test, X = matrix.test, SL.library = c("SL.glmnet","SL.randomForest","SL.gbm.onecore")) 
  model.pred <- predict(model, newdata=matrix.train)
  y.test1 <- model.pred$library.predict[,1]
  y.test2 <- model.pred$library.predict[,2]
  y.test3 <- model.pred$library.predict[,3]
  y.test4 <- model.pred$pred

  df1 <- data.frame(cbind(x.train, A=1))
  names(df1) <- names(df)
  model.pred1 <- predict(model, newdata=df1)
  y1.test1 <- model.pred1$library.predict[,1]
  y1.test2 <- model.pred1$library.predict[,2]
  y1.test3 <- model.pred1$library.predict[,3]
  y1.test4 <- model.pred1$pred

  df0 <- data.frame(cbind(x.train, A=0))
  names(df0) <- names(df)
  model.pred0 <- predict(model, newdata=df0)
  y0.test1 <- model.pred0$library.predict[,1]
  y0.test2 <- model.pred0$library.predict[,2]
  y0.test3 <- model.pred0$library.predict[,3]
  y0.test4 <- model.pred0$pred

  Amodel <- SuperLearner(Y = A.test, X = Amatrix.test, family = binomial(), SL.library = c("SL.glmnet","SL.randomForest","SL.gbm.onecore"))
  Amodel.pred <- predict(Amodel, newdata=Amatrix.train)
  A.test1 <- Amodel.pred$library.predict[,1]
  A.test2 <- Amodel.pred$library.predict[,2]
  A.test3 <- Amodel.pred$library.predict[,3]
  A.test4 <- Amodel.pred$pred

  temp2.1 <- mean( (y.train-y.test1)*(-1)^(1-A.train)/((A.test1)^A.train*(1-A.test1)^(1-A.train)) + y1.test1 - y0.test1)
  temp2.2 <- mean( (y.train-y.test2)*(-1)^(1-A.train)/((A.test2)^A.train*(1-A.test2)^(1-A.train)) + y1.test2 - y0.test2)
  temp2.3 <- mean( (y.train-y.test3)*(-1)^(1-A.train)/((A.test3)^A.train*(1-A.test3)^(1-A.train)) + y1.test3 - y0.test3)
  temp2.4 <- mean( (y.train-y.test4)*(-1)^(1-A.train)/((A.test4)^A.train*(1-A.test4)^(1-A.train)) + y1.test4 - y0.test4)

  tempvar2.1 <- var((y.train-y.test1)*(-1)^(1-A.train)/((A.test1)^A.train*(1-A.test1)^(1-A.train)) + y1.test1 - y0.test1 - temp2.1 )/length(y.train)
  tempvar2.2 <- var((y.train-y.test2)*(-1)^(1-A.train)/((A.test2)^A.train*(1-A.test2)^(1-A.train)) + y1.test2 - y0.test2 - temp2.2 )/length(y.train)
  tempvar2.3 <- var((y.train-y.test3)*(-1)^(1-A.train)/((A.test3)^A.train*(1-A.test3)^(1-A.train)) + y1.test3 - y0.test3 - temp2.3 )/length(y.train)
  tempvar2.4 <- var((y.train-y.test4)*(-1)^(1-A.train)/((A.test4)^A.train*(1-A.test4)^(1-A.train)) + y1.test4 - y0.test4 - temp2.4 )/length(y.train)

  psi.lasso[q] <- (temp1.1+temp2.1)/2
  psi.rf[q] <- (temp1.2+temp2.2)/2
  psi.gbm[q] <- (temp1.3+temp2.3)/2
  psi.sup[q] <- (temp1.4+temp2.4)/2

  psi.lasso.lower[q] <- psi.lasso[q] - qnorm(0.975)*sqrt(tempvar1.1+tempvar2.1)/2
  psi.rf.lower[q] <- psi.rf[q] - qnorm(0.975)*sqrt(tempvar1.2+tempvar2.2)/2
  psi.gbm.lower[q] <- psi.gbm[q] - qnorm(0.975)*sqrt(tempvar1.3+tempvar2.3)/2
  psi.sup.lower[q] <- psi.sup[q] - qnorm(0.975)*sqrt(tempvar1.4+tempvar2.4)/2

  psi.lasso.upper[q] <- psi.lasso[q] + qnorm(0.975)*sqrt(tempvar1.1+tempvar2.1)/2
  psi.rf.upper[q] <- psi.rf[q] + qnorm(0.975)*sqrt(tempvar1.2+tempvar2.2)/2
  psi.gbm.upper[q] <- psi.gbm[q] + qnorm(0.975)*sqrt(tempvar1.3+tempvar2.3)/2
  psi.sup.upper[q] <- psi.sup[q] + qnorm(0.975)*sqrt(tempvar1.4+tempvar2.4)/2

  ######################## TMLE & CV-TMLE & C-TMLE ######################
  out.tmle<-tmle(Y=y,A=A,W=x,cvQinit=FALSE,Q.SL.library=c("SL.glmnet","SL.randomForest","SL.gbm.onecore"),g.SL.library=c("SL.glmnet","SL.randomForest","SL.gbm.onecore"))
  psi.tmle0[q] <- out.tmle$estimates$ATE$psi
  psi.tmle0.lower[q] <- out.tmle$estimates$ATE$CI[1]
  psi.tmle0.upper[q] <- out.tmle$estimates$ATE$CI[2]

  out.tmle<-tmle(Y=y,A=A,W=x,V=nsplit,Q.SL.library=c("SL.glmnet","SL.randomForest","SL.gbm.onecore"),g.SL.library=c("SL.glmnet","SL.randomForest","SL.gbm.onecore"))
  psi.tmle1[q] <- out.tmle$estimates$ATE$psi
  psi.tmle1.lower[q] <- out.tmle$estimates$ATE$CI[1]
  psi.tmle1.upper[q] <- out.tmle$estimates$ATE$CI[2]


  folds <-by(sample(1:n,n), rep(1:nsplit, length=n), list)
  gn_seq <- build_gn_seq(A = A, W = x, SL.library = c("SL.glmnet","SL.randomForest","SL.gbm.onecore"), folds = folds)

  df <- data.frame(x, A)
  model <- SuperLearner(Y = y, X = df, SL.library = c("SL.glmnet","SL.randomForest","SL.gbm.onecore")) 
  df1 <- data.frame(cbind(x, 1))
  names(df1) <- names(df)
  Q1 <- predict(model, newdata = df1)
  df0 <- data.frame(cbind(x, 0))
  names(df0) <- names(df)
  Q0 <- predict(model, newdata = df0)
  Q <-  cbind(Q0$pred,Q1$pred)
  model.ctmle <- ctmleGeneral(Y = y, A = A, W = x, Q = Q, ctmletype = 1, gn_candidates = gn_seq$gn_candidates,  
                            gn_candidates_cv = gn_seq$gn_candidates_cv, folds = gn_seq$folds, V = length(folds))
  psi.tmle2[q] <- model.ctmle$est
  psi.tmle2.lower[q] <- model.ctmle$CI[1]
  psi.tmle2.upper[q] <- model.ctmle$CI[2]

  ######################## split data #################
  tempvar <- array(NA, c(nsplit, K, K))
  n.test.vec <- rep(NA,nsplit)
  train_ind.vec <- array(NA, c(nsplit, n))
  for (s in 1:nsplit){
    train_ind <- rbinom(n,1,0.5)
    train_ind.vec[s,] <-  train_ind
    
    x.train <- x[train_ind==0,]
    A.train <- A[train_ind==0]
    y.train <- y[train_ind==0,]
    
    x.test <- x[train_ind==1, ]
    A.test <- A[train_ind==1]
    y.test <- y[train_ind==1, ]
    n.train <- length(A.train)
    n.test <- length(A.test) 
    n.test.vec[s] <- n.test
    # fit the model, estimate the parameters for each pair(k,i)
    df <- data.frame(x, A)
    matrix.train <- data.frame(x.train, A=A.train)
    matrix.test <- data.frame(x.test, A=A.test)
    model <- SuperLearner(Y = y.train, X = matrix.train, SL.library = c("SL.glmnet","SL.randomForest","SL.gbm.onecore")) 
    model.pred <- predict(model, newdata=matrix.test)
    y.test1 <- model.pred$library.predict[,1]
    y.test2 <- model.pred$library.predict[,2]
    y.test3 <- model.pred$library.predict[,3]
    y.test4 <- model.pred$pred
    
    df1 <- data.frame(cbind(x.test, A=1))
    names(df1) <- names(df)
    model.pred1 <- predict(model, newdata=df1)
    y1.test1 <- model.pred1$library.predict[,1]
    y1.test2 <- model.pred1$library.predict[,2]
    y1.test3 <- model.pred1$library.predict[,3]
    y1.test4 <- model.pred1$pred
    
    df0 <- data.frame(cbind(x.test, A=0))
    names(df0) <- names(df)
    model.pred0 <- predict(model, newdata=df0)
    y0.test1 <- model.pred0$library.predict[,1]
    y0.test2 <- model.pred0$library.predict[,2]
    y0.test3 <- model.pred0$library.predict[,3]
    y0.test4 <- model.pred0$pred
    
    Amatrix.train <- data.frame(x.train)
    Amatrix.test <- data.frame(x.test)
    Amodel <- SuperLearner(Y = A.train, X = Amatrix.train, family = binomial(), SL.library = c("SL.glmnet","SL.randomForest","SL.gbm.onecore"))
    Amodel.pred <- predict(Amodel, newdata=Amatrix.test)
    A.test1 <- Amodel.pred$library.predict[,1]
    A.test2 <- Amodel.pred$library.predict[,2]
    A.test3 <- Amodel.pred$library.predict[,3]
    A.test4 <- Amodel.pred$pred
  
   ############### calculate psi for K*K models #############
  for (i in 1:K){
    for (j in 1:K){
      new.y0 <- eval(parse(text=paste("y0.test",toString(j),sep="")))
      new.y1 <- eval(parse(text=paste("y1.test",toString(j),sep="")))
      new.y <- eval(parse(text=paste("y.test",toString(j),sep="")))
      new.pi <- eval(parse(text=paste("A.test",toString(i),sep="")))
      psi.est[s,i,j] <- mean( (y.test-new.y)*(-1)^(1-A.test)/((new.pi)^(A.test)*(1-new.pi)^(1-A.test)) + new.y1 - new.y0)
      tempvar[s,i,j] <- var((y.test-new.y)*(-1)^(1-A.test)/((new.pi)^A.test*(1-new.pi)^(1-A.test)) + new.y1 - new.y0 - psi.est[s,i,j] )/length(y.test)
    }  
  }  
    
    for (i in 1:K){
      for (j in 1:K){
      # for fixed i
        for(k in 1:K){
          bias.i[k,i,j] <-  bias.i[k,i,j] + (psi.est[s,i,k]-psi.est[s,i,j])^2
        }
      # for fixed j
        for(k in 1:K){
          bias.j[k,i,j] <- bias.j[k,i,j] + (psi.est[s,k,j]-psi.est[s,i,j])^2
        }
      }  # end j
    } # end i
  
}   # END SPLIT

bias.i = bias.i/nsplit
bias.j = bias.j/nsplit

   #max procedure
for (i in 1:K){
  for (j in 1:K){
    bias.ii[i,j] <- max(bias.i[,i,j])
    bias.jj[i,j] <- max(bias.j[,i,j])
    
    psi.avg[,i,j] <- mean(psi.est[,i,j]) 
  }
}   

for (i in 1:K){
  for (j in 1:K){
    bias1[,i,j] <- max(bias.i[,i,j],bias.j[,i,j])
    bias2[,i,j] <- max(bias.ii[i,]) + max(bias.jj[,j])
  }
}    
    
  #min procedure
 select1[q,] <- arrayInd(which.min(bias1),dim(bias1))[2:3]
 select2[q,] <- arrayInd(which.min(bias2),dim(bias2))[2:3]

 ############################# get psi  ############################ 
 psi.save[q,,] <- psi.avg[,,]
 psi.select1[q] <- psi.avg[,select1[q,][1],select1[q,][2]]
 psi.select2[q] <- psi.avg[,select2[q,][1],select2[q,][2]]
 
 ############################ get psi CI ###########################
 cond.select1.lower[q] <- psi.est[2,select1[q,][1],select1[q,][2]] - qnorm(0.975)*sqrt(tempvar[2,select1[q,][1],select1[q,][2]])
 cond.select2.lower[q] <- psi.est[2,select2[q,][1],select2[q,][2]] - qnorm(0.975)*sqrt(tempvar[2,select2[q,][1],select2[q,][2]])
 psi.select1.lower[q] <- psi.select1[q] - qnorm(0.975)*sqrt(tempvar[1,select1[q,][1],select1[q,][2]]+tempvar[2,select1[q,][1],select1[q,][2]]+sum(colSums(train_ind.vec)==2)*(tempvar[1,select1[q,][1],select1[q,][2]]*n.test.vec[1]+tempvar[2,select1[q,][1],select1[q,][2]]*n.test.vec[2])/(2*n.test.vec[1]*n.test.vec[2]))/2
 psi.select2.lower[q] <- psi.select2[q] - qnorm(0.975)*sqrt(tempvar[1,select2[q,][1],select2[q,][2]]+tempvar[2,select2[q,][1],select2[q,][2]]+sum(colSums(train_ind.vec)==2)*(tempvar[1,select2[q,][1],select2[q,][2]]*n.test.vec[1]+tempvar[2,select2[q,][1],select2[q,][2]]*n.test.vec[2])/(2*n.test.vec[1]*n.test.vec[2]))/2
   
 cond.select1.upper[q] <- psi.est[2,select1[q,][1],select1[q,][2]] + qnorm(0.975)*sqrt(tempvar[2,select1[q,][1],select1[q,][2]])
 cond.select2.upper[q] <- psi.est[2,select2[q,][1],select2[q,][2]] + qnorm(0.975)*sqrt(tempvar[2,select2[q,][1],select2[q,][2]])
 psi.select1.upper[q] <- psi.select1[q] + qnorm(0.975)*sqrt(tempvar[1,select1[q,][1],select1[q,][2]]+tempvar[2,select1[q,][1],select1[q,][2]]+sum(colSums(train_ind.vec)==2)*(tempvar[1,select1[q,][1],select1[q,][2]]*n.test.vec[1]+tempvar[2,select1[q,][1],select1[q,][2]]*n.test.vec[2])/(2*n.test.vec[1]*n.test.vec[2]))/2
 psi.select2.upper[q] <- psi.select2[q] + qnorm(0.975)*sqrt(tempvar[1,select2[q,][1],select2[q,][2]]+tempvar[2,select2[q,][1],select2[q,][2]]+sum(colSums(train_ind.vec)==2)*(tempvar[1,select2[q,][1],select2[q,][2]]*n.test.vec[1]+tempvar[2,select2[q,][1],select2[q,][2]]*n.test.vec[2])/(2*n.test.vec[1]*n.test.vec[2]))/2
 
print(q)
} #end q

save(select1, select2, psi.save, psi.lasso, psi.rf, psi.gbm, psi.sup, psi.tmle0, psi.tmle1, psi.tmle2, psi.select1, psi.select2, file = "1.RData")

save(cond.select1.lower, cond.select2.lower, psi.select1.lower, psi.select2.lower, psi.gbm.lower, psi.rf.lower, psi.lasso.lower, psi.sup.lower, psi.tmle0.lower, psi.tmle1.lower, psi.tmle2.lower, cond.select1.upper, cond.select2.upper, psi.select1.upper, psi.select2.upper, psi.gbm.upper, psi.rf.upper, psi.lasso.upper, psi.sup.upper, psi.tmle0.upper, psi.tmle1.upper, psi.tmle2.upper, file = "b1.RData")

}
