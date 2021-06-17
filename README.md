# FAKE_News_Detection

## INTRODUCTION

Fake news causes many problems in society and the consequences are catastrophic. We can look at twitter as an example who are trying very hard to track and label fake news. We will use machine learning algorithms to perform classification or prediction of news being true otr false. We will use NLP(Natural Language Processing which is a branch of AI that deals with interaction between humans and computers using natural language. This project, with the help of machine learning algorithms performs classification/prediction of news as true and false. I will show how good cleaning techniques on the data can impact the performance of the fake news classifier. We use text-preprocessing techniques like removing stop-words, lemmatization, tokenization, and vectorization before we feed the data to the machine learning models. We have fed data to various models and compared their performance. These data cleaning techniques fall under Natural Language Processing.

Natural Language Processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and humans using the natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of the human languages in a manner that is valuable. It is the ability of a computer program to understand human language, spoken and written 

## DATASET 
The dataset is a collection of about 20800 news articles. This dataset has been compiled and created by the University of Tennessee’s Machine Learning Club, USA.
This dataset is freely available here : https://www.kaggle.com/c/fake-news/data.
The dataset consists of the following attributes:

- **id**: unique id for a news article
- **title**: the title of a news article
- **author**: author of the news article
- **text**: the text of the article; could be incomplete
- **label**: a label that marks the article as potentially unreliable
  - 1: Fake News 
  - 0: True News
 
There are 68332444 total words and 742 unique words in the dataset 

## DATA PRE-PROCESSING
### 1. Missing Data Imputation
- Identify and replace missing values for each column in the input data prior to modelling  the prediction task.
- I checked for null values in dataset.
- Used Regex  to remove special characters
- Tokenization of data
- Removing stop-words by using NLTK library
- Lemmitization
- Count Vectorization
- TF-IDF Transformation
- Checked for top 20 unigrams and bigrams before and after removing stop words
- Most common unigram in text and seperating it between reliable and unreliable news

I have applied models with 2 approaches to data cleaning:

• **Approach 1**: In the first approach, I have selected only one feature i.e. the attribute-news text and have directly applied feature extraction tools like TF-IDF vectorization after removing punctuation marks and elminating the rows with null values from the text.

• **Approach 2**: In the second approach, I have followed the steps of

  - Combining all attributes including “author”, “text” and “title” into one column.
  - Replacing null values with spaces(missing data imputation).
  - Removing stop-words and special characters.
  - Lemmatization, and finally converting.
  - Count Vectorization and TF-IDF Transformation. 
 
The difference in the results of these approaches indicates how important good NLP techniques are and how cleaning techniques like lemmatization and removal of stop words can impact the performance of machine learning models.

## Models applied 

1. Passive Aggressive Classifier  
2. Multi Layer Perceptron
4. Logistic Regression
5. Multinomial Naïve Bayes
6. Decision Tree
7. Gradient Boosting Classifier
8. Random Forest Classifier
9. K-Nearest Neighbours
10. Support Vector Machine-Linear Kernel
11. Ada Boost
12. XG Boost

All these models were applied for both approaches and compared to decide upon the more suitable Machine Learning algorithms to apply for Fake news detection and also find the models that may not be very well suited for fake news detection. 

## Results 
Accuracy is the percentage of correct predictions on the test data

Accuracy1 - Accuracy of model in firt approach<br>
Accuarcy2 - Accuracy of model in second approach

Taking average of the accuracies over the two different approaches I got the best accuracy in 
- Passive aggressive classifier: accuracy1 = 97%, accuracy2 = 98% 
- Support Vector Machine-Linear Kernel: accuracy1 = 96%, accuracy2 = 98% 

The accuracy, recall, precision, F1 score and confusion matrix for all the models is in the jupyter notebooks which can be referred

## Conclusion
Classifying news manually requires in-depth knowledge of the domain and expertise in identifying anomalies in the text. I have classified fake news articles using 11 machine learning models. This Fake News Detection tries to identify patterns in text that differentiate fake articles from true news. I extracted different textual features from the articles using Natural Language Processing for text preprocessing and also, Feature Extraction tools like 'CountVectorizer' and 'TF-IDF Transformer' and used the feature set as an input to the models. Some models have achieved comparatively higher accuracy than others. I used multiple performance metrics to compare the results for each algorithm. A Fake News Classifier should essentially have at least the following measures:
  1. High **accuracy**
  2. The number of **False Negatives** must be **minimum**. Confusion matrix will help us in finding the number of **False Negatives**. The value of False Negative indicates how many actually Fake News has been classified/predicted as Real news by the model. Clearly, this situation is not desirable because the results of fake news classified as true news may be catastrophic.

I have made some conclusions at the end of my roject:

- 10 out of 11 models showed better accuracy, recall, precision and f1- score in the second approach. 9 out of 11 models showed lower number of false negatives in the second approach. This implies that processes like removal of stop words, lemmatization and inclusion of all attributes do significantly impact performance of a machine learning model of a fake news classifier.

- I conclude that Passive Aggressive Classifier, Logistic Regression, Gradient Boosting Classifier, and SVM models show the best performance with respect to accuracy (98%,98%,97%,98%), recall (98%,98%,97%,98%), precision (98%,98%,97%,98%), f1-score (98%,98%,97%,98%) and false negative values. They exhibit relatively higher values of accuracy
with relatively lower values of false negatives (49,44,56,43). Hence, these models are better choices for the sake of fake news classification.

- KNN scores an accuracy of 66% along with 147 false negatives as per the first approach. Despite increase in its accuracy in the second approach to 85%, the number of false negative values is very high which is undesirable. Hence KNN is not an apt model for fake news classification.

- Multinomial Naive Bayes, with relatively lower accuracies of 83% and 83% in the first and second approach respectively, have significantly high false negative values of 856 and 853. Therefore Multinomial Naive Bayes is not an apt model for fake news classification.

## Running Instructions 
1. Clone this repository.
2. Download train.csv and test.csv from kaggle (https://www.kaggle.com/c/fake-news/data).
3. Create a folder called "fake-news" in the same directory as the ipynb files. 
4. Include train.csv and test.csv in the fake-news folder.
5. Run the ipynb files using jupyter notebook.
