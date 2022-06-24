# -*- coding: utf-8 -*-
"""
## 0. Installing Packages - Importing Libraries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

import re
import emoji

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.probability import FreqDist

import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""## 1. Data Preprocessing

### 1.1 Load Data
"""

#load data
train_data = pd.read_csv('train_twitter.csv')
valid_data = pd.read_csv('valid_twitter.csv')

train_data.head()

"""### 1.2 Clean Data"""

nltk.download('omw-1.4')
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


#train_df cleaning
train_data['Tweet'] = train_data['Tweet'].map(lambda x: cleaner(str(x)))
train_data = train_data.drop_duplicates()

#test_df cleaning
valid_data['Tweet'] = valid_data['Tweet'].map(lambda x: cleaner(str(x)))
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

"""## 3.Data Preparation

### 3.1 Encode Labels
"""

sentiment = []

for i in train_data["Sentiment"]:
    if i == "Positive":
        sentiment.append(1)
    else:
        sentiment.append(0)

train_data['Sentiment'] = sentiment

sentiment = []

for i in valid_data["Sentiment"]:
    if i == "Positive":
        sentiment.append(1)
    else:
        sentiment.append(0)

valid_data['Sentiment'] = sentiment

"""### 3.2 Create train and validation sets"""

#create train set
train_set = list(train_data.to_records(index=False))
train_set = [(label, word_tokenize(tweet)) for tweet, label in train_set]

#create test set
valid_set = list(valid_data.to_records(index=False))
valid_set = [(label, word_tokenize(tweet)) for tweet, label in valid_set]

"""### 3.3 Create voabulary """

#create vocabulary
index2word = ["<PAD>", "<SOS>", "<EOS>"]

for ds in [train_set, valid_set]:
    for label, tweet in ds:
        for token in tweet:
            if token not in index2word:
                index2word.append(token)

#inverse vocabulary
word2index = {token: idx for idx, token in enumerate(index2word)}

"""### 3.4 Tweets encoding and padding"""

seq_length = 40


def encode_and_pad(tweet, length):
    sos = [word2index["<SOS>"]]
    eos = [word2index["<EOS>"]]
    pad = [word2index["<PAD>"]]

    if len(tweet) < length - 2:  # -2 for SOS and EOS
        n_pads = length - 2 - len(tweet)
        encoded = [word2index[w] for w in tweet]
        return sos + encoded + eos + pad * n_pads
    else:  # tweet is longer than possible; truncating
        encoded = [word2index[w] for w in tweet]
        truncated = encoded[:length - 2]
        return sos + truncated + eos


#train set apply
train_encoded = [(encode_and_pad(tweet, seq_length), label)
                 for label, tweet in train_set]

#test set apply
valid_encoded = [(encode_and_pad(tweet, seq_length), label)
                 for label, tweet in valid_set]

"""### 3.5 Create Pytorch Dataset and DataLoaders"""

batch_size = 50

train_x = np.array([tweet for tweet, label in train_encoded])
train_y = np.array([label for tweet, label in train_encoded])
valid_x = np.array([tweet for tweet, label in valid_encoded])
valid_y = np.array([label for tweet, label in valid_encoded])

train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_ds = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))


train_dl = DataLoader(train_ds, shuffle=True,
                      batch_size=batch_size, drop_last=True)
valid_dl = DataLoader(valid_ds, shuffle=True,
                      batch_size=batch_size, drop_last=True)

"""## 4. Deep Learning

### 4.1 Model's Architecture
"""


class BiLSTM_SentimentAnalysis(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim

        # The embedding layer takes the vocab size and the embeddings size as input
        # The embeddings size is up to you to decide, but common sizes are between 50 and 100.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM layer takes in the the embedding size and the hidden vector size.
        # The hidden dimension is up to you to decide, but common values are 32, 64, 128
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # We use dropout before the final layer to improve with regularization
        self.dropout = nn.Dropout(dropout)

        # The fully-connected layer takes in the hidden dim of the LSTM and
        #  outputs a a 3x1 vector of the class scores.
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state
        """

        # The input is transformed to embeddings by passing it to the embedding layer
        embs = self.embedding(x)

        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        out, hidden = self.lstm(embs, hidden)

        # Dropout is applied to the output and fed to the FC layer
        out = self.dropout(out)
        out = self.fc(out)

        # We extract the scores for the final hidden state since it is the one that matters.
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim))


"""### 4.2 Training Procedure"""

model = BiLSTM_SentimentAnalysis(len(word2index), 64, 32, 0.2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

epochs = 20
batch_size = 50

epoch_tr_loss, epoch_vl_loss = [], []
epoch_tr_acc, epoch_vl_acc = [], []

valid_loss_min = np.Inf

for epoch in range(epochs):

    train_losses = []
    train_acc = 0.0
    model.train()

    h0, c0 = model.init_hidden(batch_size)

    h0 = h0.to(device)
    c0 = c0.to(device)

    train_batch_acc = []
    for batch_idx, batch in enumerate(train_dl):

        input = batch[0].to(device)
        target = batch[1].to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out, hidden = model(input, (h0, c0))
            #calculate the loss
            loss = criterion(out, target)
            train_losses.append(loss.item())
            #calculate accuracy
            _, preds = torch.max(out, 1)
            preds = preds.to("cpu").tolist()
            train_batch_acc.append(accuracy_score(preds, target.tolist()))
            #perform backprop
            loss.backward()
            optimizer.step()

    val_losses = []
    val_acc = 0.0
    model.eval()

    val_h, val_c = model.init_hidden(batch_size)

    val_h = val_h.to(device)
    val_c = val_c.to(device)

    val_batch_acc = []
    for batch_idx, batch in enumerate(valid_dl):

        input = batch[0].to(device)
        target = batch[1].to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            out, hidden = model(input, (val_h, val_c))
            #calculate the loss
            loss = criterion(out, target)
            val_losses.append(loss.item())
            #calculate accuracy
            _, preds = torch.max(out, 1)
            preds = preds.to("cpu").tolist()
            val_batch_acc.append(accuracy_score(preds, target.tolist()))

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = sum(train_batch_acc)/len(train_batch_acc)
    epoch_val_acc = sum(val_batch_acc)/len(val_batch_acc)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(25*'==')
    print(f'Epoch {epoch+1}')
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc} val_accuracy : {epoch_val_acc}')
    if epoch_val_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min, epoch_val_loss))
        torch.save(model.state_dict(), 'lstm_model_state.bin')
        print('Model Saved!')
        valid_loss_min = epoch_val_loss

"""### 4.3 Accuracy and Loss Plots"""

fig = plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.plot(epoch_tr_acc, label='Train Acc')
plt.plot(epoch_vl_acc, label='Validation Acc')
plt.title("Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(epoch_tr_loss, label='Train loss')
plt.plot(epoch_vl_loss, label='Validation loss')
plt.title("Loss")
plt.legend()
plt.grid()

plt.show()

"""## 5. Results"""


def get_predictions(model, data_loader):
  model = model.eval()

  predictions = []
  prediction_probs = []
  real_values = []

  val_h, val_c = model.init_hidden(batch_size)

  val_h = val_h.to(device)
  val_c = val_c.to(device)

  for batch_idx, batch in enumerate(valid_dl):

      input = batch[0].to(device)
      target = batch[1].to(device)

      optimizer.zero_grad()
      with torch.set_grad_enabled(False):
          out, hidden = model(input, (val_h, val_c))

          _, preds = torch.max(out, dim=1)

          probs = F.softmax(out, dim=1)

          predictions.extend(preds)
          prediction_probs.extend(probs)
          real_values.extend(target)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()

  return predictions, prediction_probs, real_values


model = BiLSTM_SentimentAnalysis(len(word2index), 64, 32, 0.2)
model.load_state_dict(torch.load('lstm_model_state.bin'))
model.eval()

model.to(device)

y_pred, y_pred_probs, y_test = get_predictions(
  model,
  valid_dl
)

conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0, 1]))

class_names = ['Negative', 'Positive']
confusion = pd.DataFrame(conmat, index=class_names,
                         columns=['predicted_negative', 'predicted_positive'])

print("-"*80)
print('Accuracy Score: {0:.2f}%'.format(accuracy_score(y_test, y_pred)*100))
print("-"*80)
print("Confusion Matrix\n")
print(confusion)
print("-"*80)
print("Classification Report\n")
print(classification_report(y_test, y_pred, target_names=class_names))

plt.figure(figsize=(8, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
