"""
## 0. Installing Packages - Importing Libraries
"""

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import emoji
import re
import pandas as pd
import numpy as np


"""## 1. Data Preprocessing

### 1.1 Load Data
"""

#load data
train_data = pd.read_csv('train_twitter.csv')
valid_data = pd.read_csv('valid_twitter.csv')

train_data.head()

"""### 1.2 Clean Data"""

nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))

words = set(nltk.corpus.words.words())


def cleaner(tweet):

    tokens = word_tokenize(str(tweet).replace("'", "").lower())
    without_punc = [w for w in tokens if w.isalpha()]  # Remove punct
    # Remove stopwords
    without_sw = [t for t in without_punc if t not in stop_words]
    text_len = [WordNetLemmatizer().lemmatize(t)
                for t in without_sw]  # Lemmatize
    text_cleaned = [PorterStemmer().stem(w) for w in text_len]  # Stemming
    text_cleaned = " ".join(text_cleaned)

    clean_tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+",
                         "", text_cleaned)  # Remove http links
    clean_tweet = " ".join(tweet.split())
    clean_tweet = ''.join(
        c for c in tweet if c not in emoji.UNICODE_EMOJI)  # Remove Emojis
    clean_tweet = tweet.replace("#", "").replace(
        "_", " ")  # Remove hashtag sign but keep the text

    return text_cleaned


train_data['Tweet'] = train_data['Tweet'].map(lambda x: cleaner(str(x)))
valid_data['Tweet'] = valid_data['Tweet'].map(lambda x: cleaner(str(x)))

train_data = train_data.drop_duplicates()
valid_data = valid_data.drop_duplicates()

train_data

"""## 2. Exploratory Data Analysis

### 2.1 Labels Balance
"""

sns.countplot(x="Sentiment", data=train_data)

"""### 2.2 Words Frequency Distribution Plot (first 30)"""

joined_tweets = " ".join(train_data["Tweet"])
tokenized_tweets = word_tokenize(joined_tweets)
fdist = FreqDist(tokenized_tweets)
fdist.plot(30, cumulative=False)
plt.show()

"""### 2.3 Tweets Length Distribution"""

len_list = []
for tweet in train_data['Tweet']:
  token_list = word_tokenize(tweet)
  len_list.append(len(token_list))

sns.distplot(len_list)
plt.xlim([0, 75])

"""## 3. Data Preparation

### 3.1 Encode Labels
"""

#encode labels
sentiment = []

for i in train_data["Sentiment"]:
    if i == "Negative":
        sentiment.append(0)
    else:
        sentiment.append(1)

train_data['Sentiment'] = sentiment

sentiment = []

for i in valid_data["Sentiment"]:
    if i == "Negative":
        sentiment.append(0)
    else:
        sentiment.append(1)

valid_data['Sentiment'] = sentiment

"""### 3.2 Seperate Dependent and Independent Variables"""

X_train, y_train = train_data['Tweet'], train_data['Sentiment']
X_valid, y_valid = valid_data['Tweet'], valid_data['Sentiment']

"""### 3.3 Feature Exatraction
#### Here we will use the unigram bag of words method.
"""

#n-grams definition
n = 1
vt = CountVectorizer(analyzer="word", ngram_range=(n, n))
X_train_count = vt.fit_transform(X_train)
X_valid_count = vt.transform(X_valid)

"""## 4. Machine Learning Algorithms

### 4.1 Naive Bayes
"""

nb_model = MultinomialNB()
nb_model.fit(X_train_count, y_train)

nb_pred = nb_model.predict(X_valid_count)
nb_train_pred = nb_model.predict(X_train_count)

conmat = np.array(confusion_matrix(y_valid, nb_pred, labels=[0, 1]))

class_names = ['Negative', 'Positive']
confusion = pd.DataFrame(conmat, index=class_names,
                         columns=['predicted_negative', 'predicted_positive'])

print("-"*80)
print('Accuracy Score: {0:.2f}%'.format(accuracy_score(y_valid, nb_pred)*100))
print("-"*80)
print("Confusion Matrix\n")
print(confusion)
print("-"*80)
print("Classification Report\n")
print(classification_report(y_valid, nb_pred, target_names=class_names))

plt.figure(figsize=(8, 8))
sns.heatmap(confusion_matrix(y_valid, nb_pred), annot=True, fmt="d")

viz = PrecisionRecallCurve(MultinomialNB(),
                           classes=nb_model.classes_,
                           per_class=True,
                           cmap="Set1")

viz.fit(X_train_count, y_train)
viz.score(X_valid_count, y_valid)
viz.show()

"""### 4.2 ΚΝΝ """

knn = KNeighborsClassifier()
knn_model = knn.fit(X_train_count, y_train)

knn_pred = knn_model.predict(X_valid_count)
knn_train_pred = knn_model.predict(X_train_count)

conmat = np.array(confusion_matrix(y_valid, knn_pred, labels=[0, 1]))

class_names = ['Negative', 'Positive']

confusion = pd.DataFrame(conmat, index=['Negative', 'Positive'],
                         columns=['predicted_negative', 'predicted_positive'])

print("-"*80)
print('Accuracy Score: {0:.2f}%'.format(accuracy_score(y_valid, knn_pred)*100))
print("-"*80)
print("Confusion Matrix\n")
print(confusion)
print("-"*80)
print("Classification Report\n")
print(classification_report(y_valid, knn_pred, target_names=class_names))

plt.figure(figsize=(8, 8))
sns.heatmap(confusion_matrix(y_valid, knn_pred), annot=True, fmt="d")

viz = PrecisionRecallCurve(KNeighborsClassifier(),
                           classes=knn_model.classes_,
                           per_class=True,
                           cmap="Set1")
viz.fit(X_train_count, y_train)
viz.score(X_valid_count, y_valid)
viz.show()

"""### 4.3 Support Vector Machine """

svc = SVC()
svc_model = svc.fit(X_train_count, y_train)

svc_pred = svc_model.predict(X_valid_count)
svc_train_pred = svc_model.predict(X_train_count)

conmat = np.array(confusion_matrix(y_valid, svc_pred, labels=[0, 1]))

class_names = ['Neagtive', 'Positive']
confusion = pd.DataFrame(conmat, index=class_names,
                         columns=['predicted_negative', 'predicted_positive'])

print("-"*80)
print('Accuracy Score: {0:.2f}%'.format(accuracy_score(y_valid, svc_pred)*100))
print("-"*80)
print("Confusion Matrix\n")
print(confusion)
print("-"*80)
print("Classification Report\n")
print(classification_report(y_valid, svc_pred, target_names=class_names))

plt.figure(figsize=(8, 8))
sns.heatmap(confusion_matrix(y_valid, svc_pred), annot=True, fmt="d")

viz = PrecisionRecallCurve(SVC(),
                           classes=svc_model.classes_,
                           per_class=True,
                           cmap="Set1")

viz.fit(X_train_count, y_train)
viz.score(X_valid_count, y_valid)
viz.show()
