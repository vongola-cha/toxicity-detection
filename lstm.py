import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator
from torch.optim import Adam


device='cpu'
def build_vocabulary(datasets):
    for t1,t2 in zip(*datasets):
        yield t1[1].strip().split()+t2[1].strip().split()


data=pd.read_csv('balanced_train.csv')
data=data.dropna(subset=['comment_text'])
data=data[['target','comment_text']]
train, test = train_test_split(data, test_size=0.3, random_state=2048, shuffle=True)
train_dataset,test_dataset=np.array(train),np.array(test)

vocab = build_vocab_from_iterator(build_vocabulary([train_dataset,test_dataset]), min_freq=1, specials=["<UNK>"])
vocab.set_default_index(vocab["<UNK>"])

target_pipeline=lambda x:x
text_pipeline=lambda x: vocab(x.split())
# max_words=50
def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    X = [vocab(text.split()) for text in X]
    max_words=max([len(i) for i in X])
    X = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X]
    return torch.tensor(X),torch.tensor(Y)


dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, collate_fn=vectorize_batch)
testloader=DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=vectorize_batch)


embed_len = 128
hidden_dim = 128
n_layers=3

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len, bidirectional=True,hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim*2, 1)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        prev, hidden = self.rnn(embeddings, torch.randn(n_layers*2, len(X_batch), hidden_dim))
        output=self.linear(prev)
        return output[:,-1]

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.lstm = nn.LSTM(input_size=embed_len, bidirectional=True,hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim*2, 1)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        hx = torch.randn(2 * n_layers, len(X_batch), hidden_dim)
        cx = torch.randn(2 * n_layers, len(X_batch), hidden_dim)
        prev, hidden = self.lstm(embeddings, (hx,cx))
        output=self.linear(prev)
        return output[:,-1]

def CalcValLossAndAccuracy(model, loss_fn, val_loader):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [],[],[]
        for X, Y in val_loader:
            preds = model(X).view(-1)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())

        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))

def TrainModel(model, loss_fn, optimizer, train_loader, val_loader,epochs):
    count=0
    for i in range(1, epochs+1):
        losses = []
        for X, Y in tqdm(train_loader,colour='white'):
            count+=1
            Y_preds = model(X).view(-1)

            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if count%500==0:
             print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))

            if count%1000==0:
                CalcValLossAndAccuracy(model,loss_fn,val_loader)




# rnn_predictor = RNN()
lstm_predictor=LSTM()

'''
Model train
'''

learning_rate = 1e-3
epoch=1
loss_fn = nn.BCEWithLogitsLoss()


# optimizer = Adam(rnn_predictor.parameters(), lr=learning_rate)
# TrainModel(rnn_predictor, loss_fn, optimizer, dataloader, testloader, epoch)
# PATH = "rnn_1.pt"

optimizer = Adam(lstm_predictor.parameters(), lr=learning_rate)
TrainModel(lstm_predictor, loss_fn, optimizer, dataloader, testloader, epoch)
PATH = "lstm_1.pt"
torch.save(lstm_predictor, PATH)

# Load
model = torch.load(PATH)
model.eval()
def vectorize_val(batch):
    Y, X = list(zip(*batch))
    X = [vocab(text.split()) for text in X]
    max_words=max([len(i) for i in X])
    X = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X]

    Y = [vocab(text.split()) for text in Y]
    max_words = max([len(i) for i in Y])
    Y = [tokens + ([0] * (max_words - len(tokens))) if len(tokens) < max_words else tokens[:max_words] for tokens in Y]
    return torch.tensor(X),torch.tensor(Y)

Test=pd.read_csv('new_val.csv')
Test=Test[['less_toxic','more_toxic']]
Test=Test.dropna(subset=['less_toxic','more_toxic'])
test_dataset=np.array(Test)

test_loader=DataLoader(test_dataset, shuffle=False, collate_fn=vectorize_val)

preds=[]
for X, Y in tqdm(test_loader,colour='white'):
    p1 = torch.sigmoid(model(X).view(-1)).detach().item()
    p2=torch.sigmoid(model(Y).view(-1)).detach().item()
    if p1>p2:
        preds.append(0)
    else:
        preds.append(1)
print('The average annotation score is:',1-sum(preds)/len(preds))


