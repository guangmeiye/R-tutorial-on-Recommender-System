#This code is identical to the one in the tutorial, however, more detailed explanations are provided.
#Install and load packages
options(repos = list(CRAN="http://cran.rstudio.com/"))
install.packages("recommenderlab",quiet=TRUE)
install.packages("rrecsys",quiet=TRUE)
library(devtools,quietly = TRUE)
devtools::install_github("tarashnot/SVDApproximation",quiet =TRUE)
library(recommenderlab,quietly = TRUE)
library(SVDApproximation,quietly = TRUE)
library(data.table,quietly = TRUE)
library(RColorBrewer,quietly = TRUE)
library(ggplot2,quietly = TRUE)

#Read Data, please change the filepath to the folder where you keep the Data.csv
ratings <- read.csv("/Users/miaohu/Downloads/Data.csv")

#Convert the data to sparseMatrix class.
sparse_ratings <- sparseMatrix(i = ratings$user, j = ratings$item, x = ratings$rating, 
                               dims = c(length(unique(ratings$user)),
                                        length(unique(ratings$item))),
                               dimnames = list(paste("u", 
                               1:length(unique(ratings$user)), 
                               sep = ""),paste("m",1:length(unique(ratings$item)),
                               sep = "")))

sparse_ratings[1:10, 1:10]

#Convert the data from sparseMatrix to realRatingMatrix for recommenderlab package
real_ratings <- new("realRatingMatrix", data = sparse_ratings)
real_ratings

#Present the first 10 entries of our data
print(head(ratings,10))

#Visulization of the Data
visualize_ratings(ratings_table = ratings)




#Split the data
set.seed(1)
e <- evaluationScheme(real_ratings, method="split", train=0.8, given=-5)
time1 <- Sys.time()

#Model1 Popular 5 Examples
model <- Recommender(real_ratings, method = "POPULAR", param=list(normalize = "center"))
prediction <- predict(model, real_ratings[1:5], type="ratings")
as(prediction, "matrix")[,1:5]

#Model1 Popular 5 Full Data
model <- Recommender(getData(e, "train"), "POPULAR")
prediction <- predict(model, getData(e, "known"), type="ratings")
time2 <- Sys.time()

#Model1 RMSE and Computation Time 
rmse_popular <- calcPredictionAccuracy(prediction, getData(e, "unknown"))[1]
print(time2-time1)
rmse_popular

#Model2 UBCF 5 Examples
model <- Recommender(real_ratings, method = "UBCF", 
                     param=list(normalize = "center", method="Cosine", nn=100))

prediction <- predict(model, real_ratings[1:5], type="ratings")
as(prediction, "matrix")[,1:5]

#Model2 UBCF 5 Full Data
set.seed(1)
time3 <- Sys.time()

model <- Recommender(getData(e, "train"), method = "UBCF", 
                     param=list(normalize = "center", method="Cosine", nn=100))

prediction <- predict(model, getData(e, "known"), type="ratings")

time4 <- Sys.time()

#Mode2 RMSE and Computation Time 
rmse_ubcf <- calcPredictionAccuracy(prediction, getData(e, "unknown"))[1]

print(time4-time3)
rmse_ubcf

#Model3 IBCF 5 Examples
model <- Recommender(real_ratings, method = "IBCF", 
                       param=list(normalize = "center", method="Cosine", k=50))

prediction <- predict(model, real_ratings[1:5], type="ratings")
as(prediction, "matrix")[,1:5]

#Model3 IBCF Full Data
set.seed(1)
time5 <- Sys.time()

model <- Recommender(getData(e, "train"), method = "IBCF", 
                     param=list(normalize = "center", method="Cosine", k=50))

prediction <- predict(model, getData(e, "known"), type="ratings")
time6 <- Sys.time()

#Mode3 RMSE and Computation Time 
rmse_ibcf <- calcPredictionAccuracy(prediction, getData(e, "unknown"))[1]

print(time6-time5)
rmse_ibcf

#Model4 SVD 5 Examples
model <- Recommender(real_ratings, method = "SVD", 
                     param=list(normalize = "center", k=50))

prediction <- predict(model, real_ratings[1:5], type="ratings")
as(prediction, "matrix")[,1:5]

#Model4 SVD Full Data
time7 <- Sys.time()

model <- Recommender(getData(e, "train"), method = "SVD", 
                     param=list(normalize = "center", k=50))

prediction <- predict(model, getData(e, "known"), type="ratings")
time8 <- Sys.time()

#Model4 RMSE and Computation Time 
rmse_svd <- calcPredictionAccuracy(prediction, getData(e, "unknown"))[1]

print(time8-time7)
rmse_svd

#Detach conflicted package
detach(package:recommenderlab)
library(rrecsys,quietly = TRUE)

#Model5 FunkSVD Full Data
set.seed(1)

ratings_matrix <- as(real_ratings,"matrix")

d <- defineData(ratings_matrix)
e <- evalModel(d, folds = 1)
mf_model <- evalPred(e, "funk", k = 10, steps = 100, regCoef = 0.0001, learningRate = 0.001, biases = F)

#Model5 RMSE and Computation Time 
rmse_funkSVD <- mf_model$RMSE[1]
timedif1 <- mf_model$Time[1]
rmse_funkSVD

#Model6 User-Based KNN Full Data
set.seed(1)

d <- defineData(ratings_matrix)
e <- evalModel(d, folds = 1)
mf_model <- evalPred(e, "ubknn", simFunct = "Pearson", neigh = 5)
#Model6 RMSE and Computation Time 
rmse_ubknn <- mf_model$RMSE[1]
timedif2 <- mf_model$Time[1]
rmse_ubknn

#Summary Table
modelname <- c("Popular","UBCF","IBCF","SVD", "funkSVD","User-Based KNN")
time <- c(time2-time1,time4-time3,time6-time5,time8-time7,timedif1,timedif2)
RMSE <- c(rmse_popular,rmse_ubcf,rmse_ibcf,rmse_svd,rmse_funkSVD,rmse_ubknn)

comparison <- data.frame(Model = modelname,ProcessingTime = time,RMSE = RMSE)
print(comparison)

#Summary Visualization
ggplot(comparison, aes(as.numeric(ProcessingTime), RMSE, colour = Model)) + 
  geom_point(size=3)+
  xlab("Processing Time")+
  geom_line(aes(y = min(RMSE)), color = "red", linetype = "dotted")
