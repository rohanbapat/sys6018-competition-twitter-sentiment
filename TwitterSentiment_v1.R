############################################################################
library(RCurl)
library(tm)
library(SnowballC)

# Import train dataset from github
trainURL <- getURL('https://raw.githubusercontent.com/rohanbapat/sys6018-competition-twitter-sentiment/master/train.csv')
train_master <- read.csv(text = trainURL)

# Create copy of imported dataset
train_df <- train_master

# -----------------------------------------------------------------------------------------------

# Data cleaning

# Remove all hashtags
train_df$text_cleaned <- sapply(train_df$text, function(x) gsub("@\\w+ *","",x))

# Remove strings starting with http

train_df$text_cleaned <- sapply(train_df$text_cleaned, paste0, " ")
train_df$text_cleaned <- sapply(train_df$text_cleaned, function(x) gsub("http?://.*?\\s", "", x))

# Retain only alphabets
train_df$text_cleaned <- sapply(train_df$text_cleaned, function(x) gsub("[^0-9A-Za-z'  ]", "",x))

# Convert to lower
train_df$text_cleaned <- tolower(train_df$text_cleaned)

# Extract words from sentences
train_df$words <- sapply(train_df$text_cleaned, strsplit, " ")

# Stem words
train_df$stemwords <- sapply(train_df$words, lapply, stemDocument)

# Unlist list of words
train_df$stemwords <- sapply(train_df$stemwords, unlist)

# -----------------------------------------------------------------------------------------------
