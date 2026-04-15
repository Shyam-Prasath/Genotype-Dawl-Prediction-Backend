###Amat and Gmat are symmetric relationship matrices with row and column names (line names). We assume Gmat contains only a subset of the lines in Gmat. w is weight of pedigree for genotyped lines.
install.packages("EMMREML")
library(EMMREML)
install.packages("rrBLUP")
library(rrBLUP)

Hmatfunc<-function(Amat, Gmat, w){

A11 <- Amat [rownames(Amat)%in%rownames(Gmat),colnames(Amat)%in%colnames(Gmat)]
G=Gmat[match(rownames(A11),rownames(Gmat)),match(rownames(A11),rownames(Gmat))]
G=G/mean(diag(G))*mean(diag(A11))

invA11=solve(A11)
Gw= (1-w)*G+w*A11
A22=Amat[!(rownames(Amat)%in%rownames(A11)),!(colnames(Amat)%in%colnames(A11))]
A12=Amat[match(rownames(G),rownames(Amat)),!(colnames(Amat)%in%colnames(A11))]

Hmat11=Gw
Hmat12=t(A12)%*%invA11%*%Gw
Hmat22=A22+t(A12)%*%invA11%*%(Gw-A11)%*%invA11%*%A12

Hmat=rbind(cbind(Hmat11,t(Hmat12)),cbind(Hmat12,Hmat22))
Hmat<-Hmat[match(rownames(Amat), rownames(Hmat)),match(colnames(Amat), colnames(Hmat))]
return(Hmat)
}


######optimization

optimumw<-function(y,X,Z,Amat,Gmat){
	library(EMMREML)
	optimizationfunc<-function(w){
	Hmatw<-Hmatfunc(Amat, Gmat, w=w)
	reml=emmreml(y=y,X=X,Z=Z,K=Hmatw)	$loglik
	return(-reml)
	}
	optout<-optimize(optimizationfunc, lower = 9^(-9), upper = 1, tol = 1e-10)
return(optout)
}


Pheno<-read.table("BLUPS_DTA_PH_EH.txt", header=T)
histogram(Pheno$EH)
K1<-read.table("K1_.txt", header=T)

K2<-read.table("K2_.txt", header=T)
K1=(K1)/2
K2=(K2)/2

Pheno$Full_name<-factor(as.character(Pheno$Full_name), levels=rownames(K2))
Hmat<-Hmatfunc(Amat=as.matrix(K2), Gmat=as.matrix(K1), w=.5)



###### START ########################
set.seed(98766)

for (i in 1:30){

Brazillines_complete<-as.character(Pheno[Pheno$Panel=="USP",1]) # All USP set
Brazillines<-as.character(Pheno[Pheno$Panel=="USP",1][sample(1:63, 32)]) # Validation set

Brazillines_previous= setdiff(Brazillines_complete,Brazillines)
Brazillines_remain= Brazillines_previous # to use intoTSG4

popcorn= read.table("popcorn.txt", h=T) #excluding popcorn lines from dataset
popcorn1=as.character(popcorn$Lines) #popcorn lines to excluded in the prediction

########################        TSG1        #############################
 
          ############              TS= 10 USP  VS= 53 USP        ##############

Brazillines1_USP<-Pheno[Pheno$Panel=="USP",1][sample(1:63, 53)]
Trainset1_USP<-setdiff(Pheno[Pheno$Panel=="USP",1],Patriclines1_USP)

Phenotrain1_USP<-Pheno[Pheno$Full_name%in%Trainset1_USP,]
Phenotest1_USP<-Pheno[Pheno$Full_name%in%Brazillines1_USP,]

Ztrain1_USP<-model.matrix(~-1+Phenotrain1_USP$Full_name)
Ztest1_USP<-model.matrix(~-1+Phenotest1_USP$Full_name)

Phenotrain1_USP$Panel<-factor(as.character(Phenotrain1_USP$Panel))
Xtrain1_USP<-matrix(1, nrow=length(Phenotrain1_USP$Panel), ncol=1)

#PH trait
mmout1_USP_PH<-mixed.solve(Phenotrain1_USP$PH,X=Xtrain1_USP,Z=Ztrain1_USP,K=as.matrix(K2))
corPH1_USP_PH<-cor(Phenotest1_USP$PH, Ztest1_USP%*%mmout1_USP_PH$u)
corPH1_USP_PH

#EH trait
mmout1_USP_EH<-mixed.solve(Phenotrain1_USP$EH,X=Xtrain1_USP,Z=Ztrain1_USP,K=as.matrix(K2))
corEH1_USP_EH<-cor(Phenotest1_USP$EH, Ztest1_USP%*%mmout1_USP_EH$u)
corEH1_USP_EH

            ###############  TS= 20 USP  VS= 43 USP ###############

Brazillines2_USP<-Pheno[Pheno$Panel=="USP",1][sample(1:63, 43)]
Trainset1_USP<-setdiff(Pheno[Pheno$Panel=="USP",1],Patriclines2_USP)

Phenotrain2_USP<-Pheno[Pheno$Full_name%in%Trainset2_USP,]
Phenotest2_USP<-Pheno[Pheno$Full_name%in%Brazillines2_USP,]

Ztrain2_USP<-model.matrix(~-1+Phenotrain2_USP$Full_name)
Ztest2_USP<-model.matrix(~-1+Phenotest2_USP$Full_name)

Phenotrain2_USP$Panel<-factor(as.character(Phenotrain2_USP$Panel))
Xtrain2_USP<-matrix(1, nrow=length(Phenotrain2_USP$Panel), ncol=1)

#PH trait
mmout2_USP_PH<-mixed.solve(Phenotrain2_USP$PH,X=Xtrain2_USP,Z=Ztrain2_USP,K=as.matrix(K2))
corPH2_USP_PH<-cor(Phenotest2_USP$PH, Ztest2_USP%*%mmout2_USP_PH$u)
corPH2_USP_PH

#EH trait
mmout2_USP_EH<-mixed.solve(Phenotrain2_USP$EH,X=Xtrain2_USP,Z=Ztrain2_USP,K=as.matrix(K2))
corPH2_USP_EH<-cor(Phenotest2_USP$EH, Ztest2_USP%*%mmout2_USP_EH$u)
corPH2_USP_EH

                 ############## TS= 30 USP  VS= 33 USP ###############

Brazillines3_USP<-Pheno[Pheno$Panel=="USP",1][sample(1:63, 33)]
Trainset3_USP<-setdiff(Pheno[Pheno$Panel=="USP",1],Patriclines3_USP)

Phenotrain3_USP<-Pheno[Pheno$Full_name%in%Trainset3_USP,]
Phenotest3_USP<-Pheno[Pheno$Full_name%in%Brazillines3_USP,]]

Ztrain3_USP<-model.matrix(~-1+Phenotrain3_USP$Full_name)
Ztest3_USP<-model.matrix(~-1+Phenotest3_USP$Full_name)

Phenotrain3_USP$Panel<-factor(as.character(Phenotrain3_USP$Panel))
Xtrain3_USP<-matrix(1, nrow=length(Phenotrain3_USP$Panel), ncol=1)

#PH trait
mmout3_USP_PH<-mixed.solve(Phenotrain3_USP$PH,X=Xtrain3_USP,Z=Ztrain3_USP,K=as.matrix(K2))
corPH3_USP_PH<-cor(Phenotest3_USP$PH, Ztest3_USP%*%mmout3_USP_PH$u)
corPH3_USP_PH

#EH trait
mmout3_USP_EH<-mixed.solve(Phenotrain3_USP$EH,X=Xtrain3_USP,Z=Ztrain3_USP,K=as.matrix(K2))
corPH3_USP_EH<-cor(Phenotest3_USP$EH, Ztest3_USP%*%mmout3_USP_EH$u)
corPH3_USP_EH


########################        TSG2        #############################

# 32 USP Lines = Validation set ---- 31 USP + NCRIPS + ASSO = Trainee set #

Trainset1<-setdiff(rownames(K2),c(popcorn1, Brazillines))

Phenotrain1<-Pheno[Pheno$Full_name%in%Trainset1,]
Phenotest1<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain1<-model.matrix(~-1+Phenotrain1$Full_name)
Ztest1<-model.matrix(~-1+Phenotest1$Full_name)
Xtrain1<-matrix(1, nrow=length(Phenotrain1$Panel), ncol=1)
Phenotrain1$Panel<-factor(as.character(Phenotrain1$Panel))

#PH trait
mmout1<-mixed.solve(Phenotrain1$PH,X=Xtrain1,Z=Ztrain1,K=as.matrix(K2))
corPH1<-cor(Phenotest1$PH, Ztest1%*%mmout1$u)
corPH1

#EH trait
mmout1.1<-mixed.solve(Phenotrain1$EH,X=Xtrain1,Z=Ztrain1,K=as.matrix(K2))
corEH1<-cor(Phenotest1$EH, Ztest1%*%mmout1.1$u)
corEH1

            #  32 USP Lines= Validation set ---- NCRIPS + ASSO = Trainee set  #

Brazillines_2<-as.character(Pheno[Pheno$Panel=="USP",1]) # including all lines to avoid cross validation on Trainingset2
(NCRIPS + ASSO)

Trainset2<-setdiff(rownames(K2),c(popcorn1, Brazillines_2))#excluding USP lines from TS
Phenotrain2<-Pheno[Pheno$Full_name%in%Trainset2,]
Phenotest2<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain2<-model.matrix(~-1+Phenotrain2$Full_name)
Ztest2<-model.matrix(~-1+Phenotest2$Full_name)
Xtrain2<-matrix(1, nrow=length(Phenotrain2$Panel), ncol=1)
Phenotrain2$Panel<-factor(as.character(Phenotrain2$Panel))

#PH trait
mmout2<-mixed.solve(Phenotrain2$PH,X=Xtrain2,Z=Ztrain2,K=as.matrix(K2))
corPH2<-cor(Phenotest2$PH, Ztest2%*%mmout2$u)
corPH2

#EH trait
mmout2.1<-mixed.solve(Phenotrain2$EH,X=Xtrain2,Z=Ztrain2,K=as.matrix(K2))
corEH2<-cor(Phenotest2$EH, Ztest2%*%mmout2.1$u)
corEH2

                   #  32 USP Lines= Validation set ---- NCRIPS = Trainee set  #

Trainset3<-setdiff(Pheno[Pheno$Panel=="NCRIPS",1], c(popcorn1,Brazillines))
Phenotrain3<-Pheno[Pheno$Full_name%in%Trainset3,]
Phenotest3<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain3<-model.matrix(~-1+Phenotrain3$Full_name)
Ztest3<-model.matrix(~-1+Phenotest3$Full_name)
Xtrain3<-matrix(1, nrow=length(Phenotrain3$Panel), ncol=1)
Phenotrain3$Panel<-factor(as.character(Phenotrain3$Panel))

#optw<-optimumw(y=Phenotrain3$PH,X=Xtrain3,Z=Ztrain3,Amat=as.matrix(K2),Gmat=as.matrix(K1)) #optimizing population set

#PH trait
mmout3<-mixed.solve(Phenotrain3$PH,X=Xtrain3,Z=Ztrain3,K=as.matrix(K2))
corPH3<-cor(Phenotest3$PH, Ztest3%*%mmout3$u)
corPH3

#EH trait
mmout3.1<-mixed.solve(Phenotrain3$EH,X=Xtrain3,Z=Ztrain3,K=as.matrix(K2))
corEH3<-cor(Phenotest3$EH, Ztest3%*%mmout3.1$u)
corEH3

                   # 32 USP Lines= Validation set ----  ASSO = Trainee set  #

Trainset4<-setdiff(Pheno[Pheno$Panel=="ASSO",1],c(popcorn1,Brazillines))
Phenotrain4<-Pheno[Pheno$Full_name%in%Trainset4,]
Phenotest4<-Pheno[Pheno$Full_name%in%Brazillines,]

Ztrain4<-model.matrix(~-1+Phenotrain4$Full_name)
Ztest4<-model.matrix(~-1+Phenotest4$Full_name)
Phenotrain4$Panel<-factor(as.character(Phenotrain4$Panel))
Xtrain4<-matrix(1, nrow=length(Phenotrain4$Panel), ncol=1)

#optw<-optimumw(y=Phenotrain$PH,X=Xtrain,Z=Ztrain,Amat=as.matrix(K2),Gmat=as.matrix(K1)) 

#PH trait
mmout4<-mixed.solve(Phenotrain4$PH,X=Xtrain4,Z=Ztrain4,K=as.matrix(K2))
corPH4<-cor(Phenotest4$PH, Ztest4%*%mmout4$u)
corPH4

#EH trait
mmout4.1<-mixed.solve(Phenotrain4$EH,X=Xtrain4,Z=Ztrain4,K=as.matrix(K2))
corEH4<-cor(Phenotest4$EH, Ztest4%*%mmout4.1$u)
corEH4


####### USP Lines= Validation set ----  NCRIPS_cluster = Trainee set  ######

Lines_cluster1=read.table("Cluster1.txt" , h=T)

Trainset5<-setdiff(Lines_cluster1[Lines_cluster1$Panel==" NCRIPS",1],Brazillines)
Phenotrain5<-Pheno[Pheno$Full_name%in%Trainset5,]
Phenotest5<-Pheno[Pheno$Full_name%in%Brazillines,]

Ztrain5<-model.matrix(~-1+Phenotrain5$Full_name)
Ztest5<-model.matrix(~-1+Phenotest5$Full_name)
Phenotrain5$Panel<-factor(as.character(Phenotrain5$Panel))
Xtrain5<-matrix(1, nrow=length(Phenotrain5$Panel), ncol=1)

#optw5<-optimumw(y=Phenotrain5$PH,X=Xtrain5,Z=Ztrain5,Amat=as.matrix(K2),Gmat=as.matrix(K1)) 


#PH trait
mmout5<-mixed.solve(Phenotrain5$PH,X=Xtrain5,Z=Ztrain5,K=as.matrix(K2))
corPH5<-cor(Phenotest5$PH, Ztest5%*%mmout5$u)
corPH5

#EH trait
mmout5.1<-mixed.solve(Phenotrain5$EH,X=Xtrain5,Z=Ztrain5,K=as.matrix(K2))
corEH5<-cor(Phenotest5$EH, Ztest5%*%mmout5.1$u)
corEH5

               #   USP Lines= Validation set ----  ASSO_cluster = Trainee set  #

Trainset6<-setdiff(Lines_cluster1[Lines_cluster1$Panel=="ASSO",1],Brazillines)

Phenotrain6<-Pheno[Pheno$Full_name%in%Trainset6,]
Phenotest6<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain6<-model.matrix(~-1+Phenotrain6$Full_name)
Ztest6<-model.matrix(~-1+Phenotest6$Full_name)
Phenotrain6$Panel<-factor(as.character(Phenotrain6$Panel))
Xtrain6<-matrix(1, nrow=length(Phenotrain6$Panel), ncol=1)

#optw6<-optimumw(y=Phenotrain6$PH,X=Xtrain6,Z=Ztrain6,Amat=as.matrix(K2),Gmat=as.matrix(K1)) 

#PH trait
mmout6<-mixed.solve(Phenotrain6$PH,X=Xtrain6,Z=Ztrain6,K=as.matrix(K2))
corPH6<-cor(Phenotest6$PH, Ztest6%*%mmout6$u)
corPH6

#EH trait
mmout6.1<-mixed.solve(Phenotrain6$EH,X=Xtrain6,Z=Ztrain6,K=as.matrix(K2))
corEH6<-cor(Phenotest6$EH, Ztest6%*%mmout6.1$u)
corEH6


########################        TSG3        #############################

                                               # Optimized TS #

#X pcs of big K
#candidates are names of ames and asso pop
#test usp
K2<-as.matrix(read.table("K2.txt", header=T))

svdK2<-svd(K2, nu=20, nv=20)
X<-K2%*%svdK2$v
rownames(X)[1:5]

candidates<-as.character(Pheno$Full_name[Pheno$Panel%in%c("AMES","ASSO")])
test=setdiff(Pheno$Full_name,candidates) # including only USP lines
candidates= setdiff(candidates,popcorn1)#excluding popcorn Lines

install.packages("STPGA")
library(STPGA)

                                      # Optimized 50 lines NCRIPS + ASSO  #

ListTrain7<-GenAlgForSubsetSelection(P=X,Candidates=candidates,Test=test,ntoselect=50, 
npop=500, nelite=5, mutprob=.8, niterations=500, lambda=1e-7,plotiters=T)

plot(X[,1],X[,2], pch=as.numeric(rownames(X)%in%ListTrain7[[1]]), col=as.numeric(rownames(X)%in%ListTrain7[[1]])+1)

Trainset7<-ListTrain7[[1]]
Phenotrain7<-Pheno[Pheno$Full_name%in%Trainset7,]
Phenotest7<-Pheno[Pheno$Full_name%in%Brazillines,]

Ztrain7<-model.matrix(~-1+Phenotrain7$Full_name)
Ztest7<-model.matrix(~-1+Phenotest7$Full_name)
Phenotrain7$Panel<-factor(as.character(Phenotrain7$Panel))
Xtrain7<-matrix(1, nrow=length(Phenotrain7$Panel), ncol=1)

#PH trait
mmout7<-mixed.solve(Phenotrain7$PH,X=Xtrain7,Z=Ztrain7,K=as.matrix(K2))
corPH7<-cor(Phenotest7$PH, Ztest7%*%mmout7$u)
corPH7

#EH trait
mmout7.1<-mixed.solve(Phenotrain7$EH,X=Xtrain7,Z=Ztrain7,K=as.matrix(K2))
corEH7<-cor(Phenotest7$PH, Ztest7%*%mmout7.1$u)
corEH7

                                 # Optimized 250 lines NCRIPS + ASSO  #

ListTrain8<-GenAlgForSubsetSelection(P=X,Candidates=candidates,Test=test,ntoselect=250, 
npop=500, nelite=5, mutprob=.8, niterations=500, lambda=1e-8,plotiters=T)

plot(X[,1],X[,2], pch=as.numeric(rownames(X)%in%ListTrain8[[1]]), col=as.numeric(rownames(X)%in%ListTrain8[[1]])+1)

Trainset8<-ListTrain8[[1]]
Phenotrain8<-Pheno[Pheno$Full_name%in%Trainset8,]
Phenotest8<-Pheno[Pheno$Full_name%in%Brazillines,]

Ztrain8<-model.matrix(~-1+Phenotrain8$Full_name)
Ztest8<-model.matrix(~-1+Phenotest8$Full_name)
Phenotrain8$Panel<-factor(as.character(Phenotrain8$Panel))
Xtrain8<-matrix(1, nrow=length(Phenotrain8$Panel), ncol=1)

#PH trait
mmout8<-mixed.solve(Phenotrain8$PH,X=Xtrain8,Z=Ztrain8,K=as.matrix(K2))
corPH8<-cor(Phenotest8$PH, Ztest8%*%mmout8$u)
corPH8

#EH trait
mmout8.1<-mixed.solve(Phenotrain8$EH,X=Xtrain8,Z=Ztrain8,K=as.matrix(K2))
corEH8<-cor(Phenotest8$PH, Ztest8%*%mmout8.1$u)
corEH8

                                   # Optimized 500 lines NCRIPS + ASSO  #

ListTrain9<-GenAlgForSubsetSelection(P=X,Candidates=candidates,Test=test,ntoselect=500, 
npop=500, nelite=5, mutprob=.9, niterations=500, lambda=1e-7,plotiters=T)

plot(X[,1],X[,2], pch=as.numeric(rownames(X)%in%ListTrain9[[1]]), col=as.numeric(rownames(X)%in%ListTrain9[[1]])+1)

Trainset9<-ListTrain9[[1]]
Phenotrain9<-Pheno[Pheno$Full_name%in%Trainset9,]
Phenotest9<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain9<-model.matrix(~-1+Phenotrain9$Full_name)
Ztest9<-model.matrix(~-1+Phenotest9$Full_name)
Phenotrain9$Panel<-factor(as.character(Phenotrain9$Panel))
Xtrain9<-matrix(1, nrow=length(Phenotrain9$Panel), ncol=1)

#PH trait
mmout9<-mixed.solve(Phenotrain9$PH,X=Xtrain9,Z=Ztrain9,K=as.matrix(K2))
corPH9<-cor(Phenotest9$PH, Ztest9%*%mmout9$u)
corPH9

#EH trait
mmout9.1<-mixed.solve(Phenotrain9$EH,X=Xtrain9,Z=Ztrain9,K=as.matrix(K2))
corEH9<-cor(Phenotest9$EH, Ztest9%*%mmout9.1$u)
corEH9

                               # Optimized 1000 lines NCRIPS + ASSO  #

ListTrain10<-GenAlgForSubsetSelection(P=X,Candidates=candidates,Test=test,ntoselect=1000, 
npop=1000, nelite=5, mutprob=.8, niterations=500, lambda=1e-7,plotiters=T)

plot(X[,1],-X[,2], pch=as.numeric(rownames(X)%in%ListTrain10[[1]]), col=as.numeric(rownames(X)%in%ListTrain10[[1]])+1)

Trainset10<-ListTrain10[[1]]
Phenotrain10<-Pheno[Pheno$Full_name%in%Trainset10,]
Phenotest10<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain10<-model.matrix(~-1+Phenotrain10$Full_name)
Ztest10<-model.matrix(~-1+Phenotest10$Full_name)
Phenotrain10$Panel<-factor(as.character(Phenotrain10$Panel))
Xtrain10<-matrix(1, nrow=length(Phenotrain10$Panel), ncol=1)

#PH trait
mmout10<-mixed.solve(Phenotrain10$PH,X=Xtrain10,Z=Ztrain10,K=as.matrix(K2))
corPH10<-cor(Phenotest10$PH, Ztest10%*%mmout10$u)
corPH10

#EH trait
mmout10.1<-mixed.solve(Phenotrain10$EH,X=Xtrain10,Z=Ztrain10,K=as.matrix(K2))
corEH10<-cor(Phenotest10$EH, Ztest10%*%mmout10.1$u)
corEH10

                                   # Optimized 1500 lines NCRIPS + ASSO  #

ListTrain11<-GenAlgForSubsetSelection(P=X,Candidates=candidates,Test=test,ntoselect=1500, 
npop=1500, nelite=5, mutprob=.8, niterations=500, lambda=1e-7,plotiters=T)

plot(X[,1],-X[,2], pch=as.numeric(rownames(X)%in%ListTrain11[[1]]), col=as.numeric(rownames(X)%in%ListTrain11[[1]])+1)

Trainset11<-ListTrain11[[1]]
Phenotrain11<-Pheno[Pheno$Full_name%in%Trainset11,]
Phenotest11<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain11<-model.matrix(~-1+Phenotrain11$Full_name)
Ztest11<-model.matrix(~-1+Phenotest11$Full_name)
Phenotrain11$Panel<-factor(as.character(Phenotrain11$Panel))
Xtrain11<-matrix(1, nrow=length(Phenotrain11$Panel), ncol=1)

#PH trait
mmout11<-mixed.solve(Phenotrain11$PH,X=Xtrain11,Z=Ztrain11,K=as.matrix(K2))
corPH11<-cor(Phenotest11$PH, Ztest11%*%mmout11$u)
corPH11

#EH trait
mmout11.1<-mixed.solve(Phenotrain11$EH,X=Xtrain11,Z=Ztrain11,K=as.matrix(K2))
corEH11<-cor(Phenotest11$EH, Ztest11%*%mmout11.1$u)
corEH11

                                                       # Random TS #

                                          # Random 50 lines NCRIPS + ASSO  #

ListTrain12<- candidates[sample(1:2434, 50)]
length(ListTrain12)

Trainset12<-ListTrain12
Phenotrain12<-Pheno[Pheno$Full_name%in%Trainset12,]
dim(Phenotrain12)
Phenotest12<-Pheno[Pheno$Full_name%in%Brazillines,]
dim(Phenotest12)
Ztrain12<-model.matrix(~-1+Phenotrain12$Full_name)
Ztest12<-model.matrix(~-1+Phenotest12$Full_name)
Phenotrain12$Panel<-factor(as.character(Phenotrain12$Panel))
Xtrain12<-matrix(1, nrow=length(Phenotrain12$Panel), ncol=1)

#PH trait
mmout12<-mixed.solve(Phenotrain12$PH,X=Xtrain12,Z=Ztrain12,K=as.matrix(K2))
corPH12<-cor(Phenotest12$PH, Ztest12%*%mmout12$u)
corPH12

#EH trait
mmout12.1<-mixed.solve(Phenotrain12$EH,X=Xtrain12,Z=Ztrain12,K=as.matrix(K2))
corEH12<-cor(Phenotest12$EH, Ztest12%*%mmout12.1$u)
corEH12

                                   # Random 250 lines NCRIPS + ASSO  #

ListTrain13<- candidates[sample(1:2434, 250)]
length(ListTrain13)

Trainset13<-ListTrain13
Phenotrain13<-Pheno[Pheno$Full_name%in%Trainset13,]
dim(Phenotrain13)
Phenotest13<-Pheno[Pheno$Full_name%in%Brazillines,]
dim(Phenotest13)
Ztrain13<-model.matrix(~-1+Phenotrain13$Full_name)
Ztest13<-model.matrix(~-1+Phenotest13$Full_name)
Phenotrain13$Panel<-factor(as.character(Phenotrain13$Panel))
Xtrain13<-matrix(1, nrow=length(Phenotrain13$Panel), ncol=1)

#PH trait
mmout13<-mixed.solve(Phenotrain13$PH,X=Xtrain13,Z=Ztrain13,K=as.matrix(K2))
corPH13<-cor(Phenotest13$PH, Ztest13%*%mmout13$u)
corPH13

#EH trait
mmout13.1<-mixed.solve(Phenotrain13$EH,X=Xtrain13,Z=Ztrain13,K=as.matrix(K2))
corEH13<-cor(Phenotest13$EH, Ztest13%*%mmout13.1$u)
corEH13

                                     # Random 500 lines NCRIPS + ASSO  #

ListTrain14<- candidates[sample(1:2434, 500)]
length(ListTrain14)

Trainset14<-ListTrain14
Phenotrain14<-Pheno[Pheno$Full_name%in%Trainset14,]
dim(Phenotrain14)
Phenotest14<-Pheno[Pheno$Full_name%in%Brazillines,]
dim(Phenotest14)
Ztrain14<-model.matrix(~-1+Phenotrain14$Full_name)
Ztest14<-model.matrix(~-1+Phenotest14$Full_name)
Phenotrain14$Panel<-factor(as.character(Phenotrain14$Panel))
Xtrain14<-matrix(1, nrow=length(Phenotrain14$Panel), ncol=1)

#PH trait
mmout14<-mixed.solve(Phenotrain14$PH,X=Xtrain14,Z=Ztrain14,K=as.matrix(K2))
corPH14<-cor(Phenotest14$PH, Ztest14%*%mmout14$u)
corPH14

#EH trait
mmout14.1<-mixed.solve(Phenotrain14$EH,X=Xtrain14,Z=Ztrain14,K=as.matrix(K2))
corEH14<-cor(Phenotest14$EH, Ztest14%*%mmout14.1$u)
corEH14

###### Random 1000 lines AMES + ASSO  #######################

ListTrain15<- candidates[sample(1:2434, 1000)]
length(ListTrain15)

Trainset15<-ListTrain15
Phenotrain15<-Pheno[Pheno$Full_name%in%Trainset15,]
dim(Phenotrain15)
Phenotest15<-Pheno[Pheno$Full_name%in%Brazillines,]
dim(Phenotest15)
Ztrain15<-model.matrix(~-1+Phenotrain15$Full_name)
Ztest15<-model.matrix(~-1+Phenotest15$Full_name)
Phenotrain15$Panel<-factor(as.character(Phenotrain15$Panel))
Xtrain15<-matrix(1, nrow=length(Phenotrain15$Panel), ncol=1)

#PH trait
mmout15<-mixed.solve(Phenotrain15$PH,X=Xtrain15,Z=Ztrain15,K=as.matrix(K2))
corPH15<-cor(Phenotest15$PH, Ztest15%*%mmout15$u)
corPH15

#EH trait
mmout15.1<-mixed.solve(Phenotrain15$EH,X=Xtrain15,Z=Ztrain15,K=as.matrix(K2))
corEH15<-cor(Phenotest15$EH, Ztest15%*%mmout15.1$u)
corEH15

                                      # Random 1500 lines AMES + ASSO  #

ListTrain16<- candidates[sample(1:2434, 1500)]

Trainset16<-ListTrain16
Phenotrain16<-Pheno[Pheno$Full_name%in%Trainset16,]
dim(Phenotrain16)
Phenotest16<-Pheno[Pheno$Full_name%in%Brazillines,]
dim(Phenotest16)
Ztrain16<-model.matrix(~-1+Phenotrain16$Full_name)
Ztest16<-model.matrix(~-1+Phenotest16$Full_name)
Phenotrain16$Panel<-factor(as.character(Phenotrain16$Panel))
Xtrain16<-matrix(1, nrow=length(Phenotrain16$Panel), ncol=1)

#PH trait
mmout16<-mixed.solve(Phenotrain16$PH,X=Xtrain16,Z=Ztrain16,K=as.matrix(K2))
corPH16<-cor(Phenotest16$PH, Ztest16%*%mmout16$u)
corPH16

#EH trait
mmout16.1<-mixed.solve(Phenotrain16$EH,X=Xtrain16,Z=Ztrain16,K=as.matrix(K2))
corEH16<-cor(Phenotest16$EH, Ztest16%*%mmout16.1$u)
corEH16

#########################       TSG4      ##############################

                        # (Optimized 50 lines NCRIPS + ASSO) + 31 USP   #

Trainset17<-c(ListTrain6[[1]],Brazillines_remain)
Phenotrain17<-Pheno[Pheno$Full_name%in%Trainset17,]
Phenotest17<-Pheno[Pheno$Full_name%in%Brazillines,]

Ztrain17<-model.matrix(~-1+Phenotrain17$Full_name)
Ztest17<-model.matrix(~-1+Phenotest17$Full_name)
Phenotrain17$Panel<-factor(as.character(Phenotrain17$Panel))
Xtrain17<-matrix(1, nrow=length(Phenotrain17$Panel), ncol=1)

#PH trait
mmout17<-mixed.solve(Phenotrain17$PH,X=Xtrain17,Z=Ztrain17,K=as.matrix(K2))
corPH17<-cor(Phenotest17$PH, Ztest17%*%mmout17$u)
corPH17

#EH trait
mmout17.1<-mixed.solve(Phenotrain17$EH,X=Xtrain17,Z=Ztrain17,K=as.matrix(K2))
corEH17<-cor(Phenotest17$PH, Ztest6%*%mmout17.1$u)
corEH17

                      # (Optimized 250 lines NCRIPS + ASSO) + 31 USP   #

Trainset18<-c(ListTrain7[[1]],Brazillines_remain)
Phenotrain18<-Pheno[Pheno$Full_name%in%Trainset18,]
Phenotest18<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain18<-model.matrix(~-1+Phenotrain18$Full_name)
Ztest18<-model.matrix(~-1+Phenotest18$Full_name)
Phenotrain18$Panel<-factor(as.character(Phenotrain18$Panel))
Xtrain18<-matrix(1, nrow=length(Phenotrain18$Panel), ncol=1)

#PH trait
mmout18<-mixed.solve(Phenotrain18$PH,X=Xtrain18,Z=Ztrain18,K=as.matrix(K2))
corPH18<-cor(Phenotest18$PH, Ztest18%*%mmout18$u)
corPH18

#EH trait
mmout18.1<-mixed.solve(Phenotrain18$EH,X=Xtrain18,Z=Ztrain18,K=as.matrix(K2))
corEH18<-cor(Phenotest18$EH, Ztest18%*%mmout18.1$u)
corEH18

                     # (Optimized 500 lines NCRIPS + ASSO) + 31 USP   #

Trainset19<-c(ListTrain8[[1]],Brazillines_remain)
Phenotrain19<-Pheno[Pheno$Full_name%in%Trainset19,]
Phenotest19<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain19<-model.matrix(~-1+Phenotrain19$Full_name)
Ztest19<-model.matrix(~-1+Phenotest19$Full_name)
Phenotrain19$Panel<-factor(as.character(Phenotrain19$Panel))
Xtrain19<-matrix(1, nrow=length(Phenotrain19$Panel), ncol=1)

#PH trait
mmout19<-mixed.solve(Phenotrain19$PH,X=Xtrain19,Z=Ztrain19,K=as.matrix(K2))
corPH19<-cor(Phenotest19$PH, Ztest19%*%mmout19$u)
corPH19

#EH trait
mmout19.1<-mixed.solve(Phenotrain19$EH,X=Xtrain19,Z=Ztrain19,K=as.matrix(K2))
corEH19<-cor(Phenotest19$EH, Ztest19%*%mmout19.1$u)
corEH19


###### (Optimized 1000 lines NCRIPS + ASSO) + 31 USP    ###############

Trainset20<-c(ListTrain9[[1]],Brazillines_remain)
Phenotrain20<-Pheno[Pheno$Full_name%in%Trainset20,]
Phenotest20<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain20<-model.matrix(~-1+Phenotrain20$Full_name)
Ztest20<-model.matrix(~-1+Phenotest20$Full_name)
Phenotrain20$Panel<-factor(as.character(Phenotrain20$Panel))
Xtrain20<-matrix(1, nrow=length(Phenotrain20$Panel), ncol=1)

#PH trait
mmout20<-mixed.solve(Phenotrain20$PH,X=Xtrain20,Z=Ztrain20,K=as.matrix(K2))
corPH20<-cor(Phenotest20$PH, Ztest20%*%mmout20$u)
corPH20

#EH trait
mmout20.1<-mixed.solve(Phenotrain20$EH,X=Xtrain20,Z=Ztrain20,K=as.matrix(K2))
corEH20<-cor(Phenotest20$EH, Ztest20%*%mmout20.1$u)
corEH20

                    # (Optimized 1500 lines NCRIPS + ASSO) + 31 USP   #

Trainset21<-c(ListTrain10[[1]],Brazillines_remain)
Phenotrain21<-Pheno[Pheno$Full_name%in%Trainset21,]
Phenotest21<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain21<-model.matrix(~-1+Phenotrain21$Full_name)
Ztest21<-model.matrix(~-1+Phenotest21$Full_name)
Phenotrain21$Panel<-factor(as.character(Phenotrain21$Panel))
Xtrain21<-matrix(1, nrow=length(Phenotrain21$Panel), ncol=1)

#PH trait
mmout21<-mixed.solve(Phenotrain21$PH,X=Xtrain21,Z=Ztrain21,K=as.matrix(K2))
corPH21<-cor(Phenotest21$PH, Ztest21%*%mmout21$u)
corPH21

#EH trait
mmout21.1<-mixed.solve(Phenotrain21$EH,X=Xtrain21,Z=Ztrain21,K=as.matrix(K2))
corEH21<-cor(Phenotest21$EH, Ztest21%*%mmout21.1$u)
corEH21

                        # (Random 50 lines NCRIPS + ASSO) + 31 USP   #

Trainset22<-c(ListTrain12[[1]],Brazillines_remain)
Phenotrain22<-Pheno[Pheno$Full_name%in%Trainset22,]
Phenotest22<-Pheno[Pheno$Full_name%in%Brazillines,]

Ztrain22<-model.matrix(~-1+Phenotrain22$Full_name)
Ztest22<-model.matrix(~-1+Phenotest22$Full_name)
Phenotrain22$Panel<-factor(as.character(Phenotrain22$Panel))
Xtrain22<-matrix(1, nrow=length(Phenotrain22$Panel), ncol=1)

#PH trait
mmout22<-mixed.solve(Phenotrain22$PH,X=Xtrain22,Z=Ztrain22,K=as.matrix(K2))
corPH22<-cor(Phenotest22$PH, Ztest22%*%mmout22$u)
corPH22

#EH trait
mmout22.1<-mixed.solve(Phenotrain22$EH,X=Xtrain22,Z=Ztrain22,K=as.matrix(K2))
corEH22<-cor(Phenotest22$PH, Ztest6%*%mmout22.1$u)
corEH22

                      # (Random 250 lines NCRIPS + ASSO) + 31 USP   #

Trainset23<-c(ListTrain13[[1]],Brazillines_remain)
Phenotrain23<-Pheno[Pheno$Full_name%in%Trainset23,]
Phenotest23<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain23<-model.matrix(~-1+Phenotrain23$Full_name)
Ztest23<-model.matrix(~-1+Phenotest23$Full_name)
Phenotrain23$Panel<-factor(as.character(Phenotrain23$Panel))
Xtrain23<-matrix(1, nrow=length(Phenotrain23$Panel), ncol=1)

#PH trait
mmout23<-mixed.solve(Phenotrain23$PH,X=Xtrain23,Z=Ztrain23,K=as.matrix(K2))
corPH23<-cor(Phenotest23$PH, Ztest23%*%mmout23$u)
corPH23

#EH trait
mmout23.1<-mixed.solve(Phenotrain23$EH,X=Xtrain23,Z=Ztrain23,K=as.matrix(K2))
corEH23<-cor(Phenotest23$EH, Ztest23%*%mmout23.1$u)
corEH23

                     # (Random 500 lines NCRIPS + ASSO) + 31 USP   #

Trainset24<-c(ListTrain14[[1]],Brazillines_remain)
Phenotrain24<-Pheno[Pheno$Full_name%in%Trainset24,]
Phenotest24<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain24<-model.matrix(~-1+Phenotrain24$Full_name)
Ztest24<-model.matrix(~-1+Phenotest24$Full_name)
Phenotrain24$Panel<-factor(as.character(Phenotrain24$Panel))
Xtrain24<-matrix(1, nrow=length(Phenotrain24$Panel), ncol=1)

#PH trait
mmout24<-mixed.solve(Phenotrain24$PH,X=Xtrain24,Z=Ztrain24,K=as.matrix(K2))
corPH24<-cor(Phenotest24$PH, Ztest24%*%mmout24$u)
corPH24

#EH trait
mmout24.1<-mixed.solve(Phenotrain24$EH,X=Xtrain24,Z=Ztrain24,K=as.matrix(K2))
corEH24<-cor(Phenotest24$EH, Ztest24%*%mmout24.1$u)
corEH24


###### (Random 1000 lines NCRIPS + ASSO) + 31 USP    ###############

Trainset25<-c(ListTrain15[[1]],Brazillines_remain)
Phenotrain25<-Pheno[Pheno$Full_name%in%Trainset25,]
Phenotest25<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain25<-model.matrix(~-1+Phenotrain25$Full_name)
Ztest25<-model.matrix(~-1+Phenotest25$Full_name)
Phenotrain25$Panel<-factor(as.character(Phenotrain25$Panel))
Xtrain25<-matrix(1, nrow=length(Phenotrain25$Panel), ncol=1)

#PH trait
mmout25<-mixed.solve(Phenotrain25$PH,X=Xtrain25,Z=Ztrain25,K=as.matrix(K2))
corPH25<-cor(Phenotest25$PH, Ztest25%*%mmout25$u)
corPH25

#EH trait
mmout25.1<-mixed.solve(Phenotrain25$EH,X=Xtrain25,Z=Ztrain25,K=as.matrix(K2))
corEH25<-cor(Phenotest25$EH, Ztest25%*%mmout25.1$u)
corEH25

                    # (Random 1500 lines NCRIPS + ASSO) + 31 USP   #

Trainset26<-c(ListTrain16[[1]],Brazillines_remain)
Phenotrain26<-Pheno[Pheno$Full_name%in%Trainset26,]
Phenotest26<-Pheno[Pheno$Full_name%in%Brazillines,]
Ztrain26<-model.matrix(~-1+Phenotrain26$Full_name)
Ztest26<-model.matrix(~-1+Phenotest26$Full_name)
Phenotrain26$Panel<-factor(as.character(Phenotrain26$Panel))
Xtrain26<-matrix(1, nrow=length(Phenotrain26$Panel), ncol=1)

#PH trait
mmout26<-mixed.solve(Phenotrain26$PH,X=Xtrain26,Z=Ztrain26,K=as.matrix(K2))
corPH26<-cor(Phenotest26$PH, Ztest26%*%mmout26$u)
corPH26

#EH trait
mmout26.1<-mixed.solve(Phenotrain26$EH,X=Xtrain26,Z=Ztrain26,K=as.matrix(K2))
corEH26<-cor(Phenotest26$EH, Ztest26%*%mmout26.1$u)
corEH26

}

