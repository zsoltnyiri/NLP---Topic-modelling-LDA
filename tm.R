### Intention

# This is a small explorative analysis of some publicly available Airbnb datasources hosted on http://insideairbnb.com/get-the-data.html, mainly focusing on using the *tm*, *ldatuning* and *topicmodelling* NLP packages.

library(data.table)
library(textcat)
library(tm)
library(Rmpfr)
library(ldatuning)
library(topicmodels)

### Preparation

## Getting the data

# The detailed reviews and listings csv-s are downloaded from the website directly. However, since they are zipped first they have to be unzipped, then read with *data.table* which is way quicker than the built in read.table on files like these.

datasources = "reviews"
root_url = "http://data.insideairbnb.com/france/ile-de-france/paris/2020-03-15/data/"

for (source in datasources){
  download.file(paste0(root_url, source, ".csv.gz"), 
                paste0(source, ".csv.gz"))
}

for(source in datasources) {
  assign(source, fread(paste0(source, ".csv.gz")), encoding = 'UTF-8')
}

## Data cleansing

# Since I took the Parisian reviews in this example, chances are there are a lot of French reviews as well (along with the standard English ones). So first the languages have to be identified for every review, as different languages call for different ways to handle them. I'll also take a smaller sample to save computation time. Set the seed for reproductability.

set.seed(69)
review_sample = sample(reviews$comments, 1000)

# Predict the language
review_lang = textcat(review_sample)
sort(table(review_lang), decreasing = T)[1:5]
# ~ 591 English reviews are present in the sample, which is the grand majority. We will use these for the analysis.

review_sample = as.data.frame(cbind(review_sample, review_lang))

# Create a subset of the English reviews
reviews_en = as.character(review_sample$review_sample[which(review_sample$review_lang == 'english')])

## Processing the texts

# Processing the texts starts by creating a corpus of the reviews. Afterwards I'll apply various data cleansing techniques (removing white spaces, punctuations etc.). I will periodicly inspect a line to demostrate the progress of the cleansing.

# Create a corpus of the reviews and process the text
reviews_corp = Corpus(VectorSource(reviews_en))
# Inspect a random line for future reference
writeLines(as.character(reviews_corp[[1]]))

# Cleanse the text

clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, stopwords("en"))
  corpus <- tm_map(corpus, stemDocument)
  return(corpus)
}

# Apply the function
clean_corp <- clean_corpus(reviews_corp)

# Check on the results
writeLines(as.character(clean_corp[[1]]))

### Complete the stem

# As of now the standard stemCompletion function does not work properly,it needs to be applied manually
stemCompletion_mod = function(x, dict = reviews_corp) {
  PlainTextDocument(stripWhitespace(paste(stemCompletion(unlist(strsplit(as.character(x)," ")),
                                                         dictionary = dict,
                                                         type = "short"),
                                          sep = "",
                                          collapse = " ")))
}

reviews_corp_comp = lapply(clean_corp, stemCompletion_mod, dict = reviews_corp)
writeLines(as.character(reviews_corp_comp[[1]]))

## After the stemcompletion is done the format has changed to large list with more attributes which is not supported by the dtm function
# Substract the main information and transform back to corpus removing the newly added stringparts and prepare to be fed to the dtm function
reviews_corp_comp = lapply(reviews_corp_comp, function (x) x['content'])
reviews_corp_comp = Corpus(VectorSource(reviews_corp_comp))
writeLines(as.character(reviews_corp_comp[[1]]))

remove_extended_reg_ex <- function(corpus){
  toSpace = content_transformer(function(x, pattern) { return (gsub(pattern, ' ', x, fixed = TRUE))})
  corpus = tm_map(corpus, toSpace, 'list(content = "')
  corpus = tm_map(corpus, toSpace, '")')
  return(corpus)
}

reviews_corp_final = remove_extended_reg_ex(reviews_corp_comp)
writeLines(as.character(reviews_corp_final[[1]]))

# Create document-term matrix
review_dtm = DocumentTermMatrix(reviews_corp_final)

# Collapse matrix by summing over columns
freq = colSums(as.matrix(review_dtm))
sort(freq, decreasing = T)[1:10] # 10 most frequent terms
length(freq) # Total number of terms

##### TOPIC MODELLING

### Estimate the number of topics to set - Parameter tuning by boosting
# Set parameters for Gibbs sampling
k = 20
burnin = 1000
iter = 1000
seed = list(2003, 5, 63)
nstart = 3
keep = 50

control = list(nstart = nstart, seed = seed, burnin = burnin, iter = iter, keep = keep)

topic_number = FindTopicsNumber(
  review_dtm,
  topics = seq(from = 2, to = k, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = control,
  mc.cores = 4L,
  verbose = T
)

FindTopicsNumber_plot(topic_number)

## By using harmonic mean following the example of Martin Ponweiser's thesis on Model Selection by Harmonic Mean (http://epub.wu.ac.at/3558/1/main.pdf)
# The log-likelihood values are determined by first fitting a model
harmonicMean = function(logLikelihoods, precision = 2000L) {
  llMed = median(logLikelihoods)
  as.double(llMed - log(mean(exp(-mpfr(logLikelihoods,
                                       prec = precision) + llMed))))
}

# Generate numerous topic models with different numbers of topics
sequ = seq(2, k, 1)
fitted_many = lapply(sequ, function(k) LDA(review_dtm, k = k, method = "Gibbs", control = control))

# Extract loglikelihoods from each topic
logLiks_many = lapply(fitted_many, function(L)  L@logLiks[-c(1:(burnin/keep))])

# Compute harmonic means using the harmonicMean function
hm_many = sapply(logLiks_many, function(h) harmonicMean(h))

# Plot the results
plot(sequ, hm_many, type = "l", main = 'Optimal Number of topics', xlab = 'Number of topics', ylab = 'Harmonic mean')

# Compute optimum number of topics
hm_many
k = sequ[which.max(hm_many)]
k

# The result is 11, which looks plausible looking at the results of the Ldaboost as well, so that's the numebr of topics I'm going to use

### Run Latent Dirichlet allocation using Gibbs sampling
lda = LDA(review_dtm, k, method = 'Gibbs', control = control)

# Docs to topics
lda_topics = as.matrix(topics(lda))
lda_topics_sum = table(lda_topics)

# Top n terms in each topic
n = 6
lda_terms = t(as.matrix(terms(lda, n)))
lda_terms = cbind(lda_terms, lda_topics_sum)

col_list = c()
for (i in seq(1:n)) {
  name = c(paste0('word_', i))
  col_list = c(col_list, name)
}

colnames(lda_terms) <- c(col_list, 'topic_freq')
lda_terms

# Probabilities associated with each topic assignment
topic_prob = lda@gamma
topic_result = cbind(topic_prob, lda_topics)
colnames(topic_result)[12+1] = 'result no.'

           