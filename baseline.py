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


# model_path='baseline_model.pkl'
# model_path='./model/baseline_model.pth'
class CSVDataset(Dataset):
    def __init__(self, path, chunksize,nb_samples):
        self.path = path
        self.chunksize = chunksize
        self.len = nb_samples // self.chunksize

    def __getitem__(self, index):
        x = next(
            pd.read_csv(
                self.path,
                skiprows=index * self.chunksize + 1,  #+1, since we skip the header
                chunksize=self.chunksize,
                names=['comment_text','target']))
        x = torch.tensor(x.data.values)
        return x

    def __len__(self):
        return self.len

device='cpu'
def build_vocabulary(datasets):
    for t1,t2 in zip(*datasets):
        yield t1[1].strip().split()+t2[1].strip().split()


data=pd.read_csv('./data/new_train.csv')
data=data.dropna(subset=['comment_text'])
data=data[['target','comment_text']]
train, test = train_test_split(data, test_size=0.3, random_state=2048, shuffle=True)
train_dataset,test_dataset=np.array(train),np.array(test)

vocab = build_vocab_from_iterator(build_vocabulary([train_dataset,test_dataset]), min_freq=1, specials=["<UNK>"])
vocab.set_default_index(vocab["<UNK>"])

target_pipeline=lambda x:x
text_pipeline=lambda x: vocab(x.split())
max_words=50
def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    X = [vocab(text.split()) for text in X]
    X = [tokens+([0]* (max_words-len(tokens))) if len(tokens)<max_words else tokens[:max_words] for tokens in X]
    return torch.tensor(X),torch.tensor(Y)


dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, collate_fn=vectorize_batch)
testloader=DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=vectorize_batch)
# for b in dataloader:
#     print(b)
#     break

embed_len = 128
hidden_dim = 64
n_layers=2

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        prev, hidden = self.rnn(embeddings, torch.randn(n_layers, len(X_batch), hidden_dim))
        output=self.linear(prev)
        return output[:,-1]

def CalcValLossAndAccuracy(model, loss_fn, val_loader):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [],[],[]
        for X, Y in val_loader:
            preds = model(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())

        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))

def TrainModel(model, loss_fn, optimizer, train_loader, val_loader,epochs=2):
    count=0
    for i in range(1, epochs+1):
        losses = []
        for X, Y in tqdm(train_loader,colour='white'):
            count+=1
            Y_preds = model(X)

            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if count%100==0:
             print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))

            if count%500==0:
                CalcValLossAndAccuracy(model,loss_fn,val_loader)




rnn_predictor = RNN()
learning_rate = 1e-3
epoch=2
loss_fn = nn.MSELoss()

optimizer = Adam(rnn_predictor.parameters(), lr=learning_rate)

baseline_net=TrainModel(rnn_predictor, loss_fn, optimizer, dataloader, testloader, epoch)

# torch.save(baseline_net,model_path)
#
# model = torch.load(model_path)
# print(model)
# CharDict = model.CharDict
# CharList = model.CharList
# print(CharList)
# print(CharDict)




