"""
## 0. Installing Packages - Importing Libraries
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd

import re
import emoji

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from collections import defaultdict

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.probability import FreqDist

import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""## 1. Data Preproccessing

### 1.1 Load Data
"""

#load data
train_data = pd.read_csv('train_twitter.csv')
valid_data = pd.read_csv('valid_twitter.csv')

train_data.head()

"""### 1.2 Data Cleaning"""

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

"""## 3. Dataset Preperation

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

"""### 3.2 Define Sequence Length

#### Based on the 2.3 plot, we can see that most tweets, after data cleaning fluctuate with the maximum length 40. However we pass the max sequence length to 50, in order to move to move in safe paths.
"""

max_len = 40

"""### 3.3 Create Pytorch Dataset"""

pre_trained_model = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(pre_trained_model)


class TweetsDataset(Dataset):

  def __init__(self, tweets, targets, tokenizer, max_len):
    self.tweets = tweets
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.tweets)

  def __getitem__(self, item):
    tweet = str(self.tweets[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      tweet,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'tweet_text': tweet,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = TweetsDataset(
    tweets=df.Tweet.to_numpy(),
    targets=df.Sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )


batch_size = 256

train_data_loader = create_data_loader(
    train_data, tokenizer, max_len, batch_size)
val_data_loader = create_data_loader(
    valid_data, tokenizer, max_len, batch_size)

"""## 4. Sentiment Classification with BERT and Hugging Face

### 4.1 Load BERT Model
"""

bert_model = BertModel.from_pretrained(pre_trained_model)

"""### 4.2 Network's Architecture"""


class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(pre_trained_model)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    output = self.drop(pooled_output)
    return self.out(output)


"""### 4.3 Training Statements Declaration"""

model = SentimentClassifier(3)
model = model.to(device)

epochs = 20

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0

  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


"""### 4.4 Training Procedure"""

history = defaultdict(list)
best_accuracy = 0

for epoch in range(epochs):

  print(f'Epoch {epoch + 1}/{epochs}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(train_data)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(valid_data)
  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state.bin')
    best_accuracy = val_acc

"""### 4.5 Training Results """

fig = plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.plot([item.cpu().numpy()
         for item in history['train_acc']], label='train accuracy')
plt.plot([item.cpu().numpy()
         for item in history['val_acc']], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot([item for item in history['train_loss']], label='train loss')
plt.plot([item for item in history['val_loss']], label='validation loss')
plt.title('Training history')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

plt.show()

"""### 4.6 Classification Report """


def get_predictions(model, data_loader):
  model = model.eval()

  texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return texts, predictions, prediction_probs, real_values


model = model = SentimentClassifier(3)
model.load_state_dict(torch.load('best_model_state.bin'))
model.eval()

model = model.to(device)

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  val_data_loader
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
