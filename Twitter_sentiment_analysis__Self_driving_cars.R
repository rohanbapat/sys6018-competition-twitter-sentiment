# Competition 3-8
# Rohan Bapat, Robert Mahoney, Pragati Shah
# rb2te, rlm4bj, pvs3vf

library(XML)
library(tidyverse)
library(RCurl)
library(tm)
library(DAAG)
library(SnowballC)

# -------------- Import training and testing data -----------------------------------------

trainURL <- getURL('https://raw.githubusercontent.com/rohanbapat/sys6018-competition-twitter-sentiment/master/train.csv')
train_master <- read.csv(text = trainURL)
nrow_train <- nrow(train_master)

testURL <- getURL('https://raw.githubusercontent.com/rohanbapat/sys6018-competition-twitter-sentiment/master/test.csv')
test_master <- read.csv(text = testURL)
nrow_test <- nrow(test_master)

# -------------- Combine training and testing text ----------------------------------------

tweet.data.frame <- append(as.character(train_master[,'text']),as.character(test_master[,'text']))
tweet.data.frame <- data.frame(tweet.data.frame)

# -------------- Text cleaning ------------------------------------------------------------

# Flag documents which contain ! or ? 
tweet.data.frame$match_exclamation <- sapply(tweet.data.frame$tweet.data.frame, function(x) grepl("!",x))
tweet.data.frame$match_qmark <- sapply(tweet.data.frame$tweet.data.frame, function(x) grepl("?",x))

# Flag documents which contain profanities
profanity_str <- c("fkk|fuck|fck|fkkk|fakk")
tweet.data.frame$match_profanity <- sapply(tweet.data.frame$tweet.data.frame, function(x) grepl(profanity_str,x))

# Remove all hashtags
tweet.data.frame$text_cleaned <- sapply(tweet.data.frame$tweet.data.frame, function(x) gsub("@\\w+ *","",x))

# Remove strings starting with http
tweet.data.frame$text_cleaned <- sapply(tweet.data.frame$text_cleaned, paste0, " ")
tweet.data.frame$text_cleaned <- sapply(tweet.data.frame$text_cleaned, function(x) gsub("http?://.*?\\s", "", x))

# Retain only alphabets
tweet.data.frame$text_cleaned <- sapply(tweet.data.frame$text_cleaned, function(x) gsub("[^A-Za-z'  ]", "",x))

# Create a copy of cleaned dataframe
tweet.data.frame_orig <- tweet.data.frame

# -------------- Create text corpus of TF-IDF matrix ------------------------------------

# Select only the cleaned text column for creating corpus
tweet.data.frame <- data.frame(tweet.data.frame$text_cleaned)

# convert this part of the data frame to a corpus object.
tweets = VCorpus(DataframeSource(tweet.data.frame))
 
# ------------- Compute TF-IDF matrix and reduce sparsity -------------------------------

# Clean up the corpus.
tweets.clean = tm_map(tweets, stripWhitespace)                          # remove extra whitespace
tweets.clean = tm_map(tweets.clean, removeNumbers)                      # remove numbers
tweets.clean = tm_map(tweets.clean, removePunctuation)                  # remove punctuation
tweets.clean = tm_map(tweets.clean, content_transformer(tolower))       # ignore case
tweets.clean = tm_map(tweets.clean, removeWords, stopwords("english"))  # remove stop words
tweets.clean = tm_map(tweets.clean, stemDocument)                       # stem all words

# Compute TF-IDF matrix
tweets.clean.tfidf = DocumentTermMatrix(tweets.clean, control = list(weighting = weightTfIdf))

# we've still got a very sparse document-term matrix. remove sparse terms at 98% threshold.
tfidf_thresh = removeSparseTerms(tweets.clean.tfidf, 0.98)  # remove terms that are absent from at least 70% of documents

# ------------- Create train and test datasets for modelling ---------------------------

# Create training data for parametric modelling
tweet_train_df <- as.matrix(tfidf_thresh)[1:nrow_train,]
tweet_train_df <- as.data.frame(tweet_train_df)

# Retrieve additional columns including flags for ?, ! and profanities
tweet.data.frame_orig_train <- tweet.data.frame_orig[1:nrow_train,]

# cbind these additional columns to TF-IDF matrix
tweet_train_df <- cbind(tweet_train_df,
                        match_exclamation = tweet.data.frame_orig_train$match_exclamation,
                        match_qmark = tweet.data.frame_orig_train$match_qmark,
                        match_profanity = tweet.data.frame_orig_train$match_profanity)

# Add sentiment column to train dataset
tweet_train_df <- cbind(tweet_train_df,train_master[,'sentiment'])

# Update column names for train dataset
colnames(tweet_train_df)<- c(colnames(tweet_train_df)[1:(length(colnames(tweet_train_df))-1)],"Sentiment")

# Creating testing set for predictions
tweet_test_df <- as.matrix(tfidf_thresh)[(nrow_train+1):nrow(tfidf_thresh),]
tweet_test_df <- as.data.frame(tweet_test_df)

tweet.data.frame_orig_test <- tweet.data.frame_orig[(nrow_train+1):nrow(tweet.data.frame_orig),]
tweet_test_df <- cbind(tweet_test_df,
                       match_exclamation = tweet.data.frame_orig_test$match_exclamation, 
                       match_qmark = tweet.data.frame_orig_test$match_qmark, 
                       match_profanity = tweet.data.frame_orig_test$match_profanity)

# ------------- Parametric model 1 - Linear regression ----------------------------------

# Build linear model
lin_model1 <- lm(Sentiment~., data = tweet_train_df)

# leave one out cross validation of linear model
kf_cv <- CVlm(data = tweet_train_df, lin_model1 ,m = 10)

# In the cross validation implementation above, the number of folds = 10
# This can be modified to nrow(tweet_train_df) to perform LEAVE ONE OUT cross validation
# loo_cv <- CVlm(data = tweet_train_df, lin_model1 ,m = nrow(tweet_train_df))

# Generate predictions using test data
lin_predictions <- predict(lin_model1, newdata = tweet_test_df)

# Scale predictions within the range 1:5
lin_predictions_scaled <- (lin_predictions-min(lin_predictions))*5/(max(lin_predictions)-min(lin_predictions))+1

# Add upper limit of 5 to predictions
lin_predictions_scaled[lin_predictions_scaled>5] <- 5

# Round predicted outcomes to get integer output
lin_predictions_scaled_rd <- round(lin_predictions_scaled)

# Convert predictions to dataframe
lin_predictions_df <- data.frame(lin_predictions_scaled_rd)

# Format predictions into given template
colnames(lin_predictions_df) <- c("sentiment")
lin_predictions_df$id <- c(1:nrow(lin_predictions_df))
lin_predictions_df <- lin_predictions_df[,c("id","sentiment")]

# Write predictions to csv
write.csv(lin_predictions_df, "sentiment_predictions_linear.csv", row.names = F)

# ------------- Paramteric model 2 - Logistic regression ---------------------------------

# One hot encode the sentiment vector
for(i in 1:5){
  x <- tweet_train_df$Sentiment
  x[x!=i] <- 0
  x[x==i] <- 1
  tweet_train_df[paste0('Sentiment',i)] <- x
}

# Remove original sentiment variable
tweet_train_df <- subset(tweet_train_df, select = -c(Sentiment))

# Rename one hot encoded variables
predicted_vars <- c('Sentiment1','Sentiment2', 'Sentiment3', 'Sentiment4', 'Sentiment5')
predictor_vars <- colnames(tweet_train_df)[!colnames(tweet_train_df) %in% predicted_vars]

# Create logistic regressions for each value of Sentiment from 1 to 5
log_model_Sentiment1 <- glm(Sentiment1 ~ ., data = tweet_train_df[,1:47], family = 'binomial')
tweet_train_df <- subset(tweet_train_df, select = -c(Sentiment1))

log_model_Sentiment2 <- glm(Sentiment2 ~ ., data = tweet_train_df[,1:47], family = 'binomial')
tweet_train_df <- subset(tweet_train_df, select = -c(Sentiment2))

log_model_Sentiment3 <- glm(Sentiment3 ~ ., data = tweet_train_df[,1:47], family = 'binomial')
tweet_train_df <- subset(tweet_train_df, select = -c(Sentiment3))

log_model_Sentiment4 <- glm(Sentiment4 ~ ., data = tweet_train_df[,1:47], family = 'binomial')
tweet_train_df <- subset(tweet_train_df, select = -c(Sentiment4))

log_model_Sentiment5 <- glm(Sentiment5 ~ ., data = tweet_train_df[,1:47], family = 'binomial')
tweet_train_df <- subset(tweet_train_df, select = -c(Sentiment5))

# Generate predictions for the one hot encoded variables
pred_Sentiment1 <- predict(log_model_Sentiment1, newdata = tweet_test_df, type = 'response')
pred_Sentiment2 <- predict(log_model_Sentiment2, newdata = tweet_test_df, type = 'response')
pred_Sentiment3 <- predict(log_model_Sentiment3, newdata = tweet_test_df, type = 'response')
pred_Sentiment4 <- predict(log_model_Sentiment4, newdata = tweet_test_df, type = 'response')
pred_Sentiment5 <- predict(log_model_Sentiment5, newdata = tweet_test_df, type = 'response')

# Combine the 5 different columns to create a single predicted column
glm_predictions_df <- data.frame(pred_Sentiment1,pred_Sentiment2,pred_Sentiment3,pred_Sentiment4,pred_Sentiment5)
glm_predictions <- apply(glm_predictions_df,1,function(x) which(x == max(x)))

# Convert predictions to dataframe
glm_predictions_df <- data.frame(glm_predictions)

# Format predictions into given template
colnames(glm_predictions_df) <- c("sentiment")
glm_predictions_df$id <- c(1:nrow(glm_predictions_df))
glm_predictions_df <- glm_predictions_df[,c("id","sentiment")]

# Write predictions to csv
write.csv(glm_predictions_df, "sentiment_predictions_glm.csv", row.names = F)

# ------------- Non- paramteric model 1 - k Nearest Neighbours ---------------------------------

tweets.99 = removeSparseTerms(tweets.clean.tfidf, 0.99)  # remove terms that are absent from at least 99% of documents (keep most terms)
tweets.99

dtm.tweets.99 = as.matrix(tweets.99)

#train_words<-colnames(dtm.tweets.99)
dtm.dist.matrix = as.matrix(dist(dtm.tweets.99))

nrow(dtm.dist.matrix)
nrow(train_master)
nrow(test_master)

#Loop over each tweet in the test set and assign it the most most common sentiment score among its n-nearest neighbors
#We found our highest prediction came with setting n=20. The distane between tweets are taken from the DocumentTermMatrix distances calculated above
#Because the test training and test tweets were combined into the same matrix the loop begins at the index after the training data ends. 

# ------------------------- kNN with k = 5 ------------------------------------------------

test_sentiment<-vector()
for (i in (nrow(train_master)+1):nrow(dtm.dist.matrix)) {
  
  # Sort each document in the order of decreasing value of distance
  most.similar.documents <- order(dtm.dist.matrix[i,], decreasing = FALSE)
  
  # Get the 5 nearest neighbours  
  most.similar.documents <- head(most.similar.documents[most.similar.documents<=nrow(train_master)],n=5)
  
  test_sentiment[i-nrow(train_master)] <- as.numeric(tail(names(sort(table(train_master[most.similar.documents,1]))), 1))
  
}

# ------------------------- kNN with k = 20 ------------------------------------------------

test_sentiment<-vector()
for (i in (nrow(train_master)+1):nrow(dtm.dist.matrix)) {
  
  # Sort each document in the order of decreasing value of distance
  most.similar.documents <- order(dtm.dist.matrix[i,], decreasing = FALSE)
  
  # Get the20 nearest neighbours  
  most.similar.documents <- head(most.similar.documents[most.similar.documents<=nrow(train_master)],n=20)
  
  test_sentiment[i-nrow(train_master)] <- as.numeric(tail(names(sort(table(train_master[most.similar.documents,1]))), 1))
  
}

# A higher value of k increases the accuracy of prediction using kNN
# Using trial and error, the optimal value of k is found at k = 20

# Format kNN predictions file
predictions<-cbind(id=test_master$id,sentiment=test_sentiment)

# Write predictions to csv
write.csv(predictions, file = "sentiment_predictions_knn.csv",row.names=FALSE)