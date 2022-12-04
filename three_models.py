import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator
from torch.optim import Adam

'''
User Parameters
'''
MODEL = 'GRU'  # 'LSTM' / 'RNN' / 'GRU'
PRE_TRAIN = True  # True / False
embed_model = 'glove'

device = 'cpu'

'''
Load data
'''
data = pd.read_csv('data/balanced_train.csv')
data = data.dropna(subset=['comment_text'])
data = data[['target', 'comment_text']]
train, test = train_test_split(data, test_size=0.3, random_state=2048, shuffle=True)
train_dataset, test_dataset = np.array(train), np.array(test)

'''
Pre-train Model: Glove
'''
if PRE_TRAIN:
    vocab, embeddings = [], []
    with open('data/glove.6B.50d.txt', 'r', encoding='utf—8') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)
    # insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    print(vocab_npa[:10])

    pad_emb_npa = np.zeros((1, embs_npa.shape[1]))  # embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)  # embedding for '<unk>' token.

    # insert embeddings for pad and unk tokens at top of embs_npa.
    embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa))
    print(embs_npa.shape)


'''
txt ---> vocabulary
'''


def build_vocabulary(datasets):
    for t1, t2 in zip(*datasets):
        yield t1[1].strip().split() + t2[1].strip().split()


vocab = build_vocab_from_iterator(build_vocabulary([train_dataset, test_dataset]), min_freq=1, specials=["<UNK>"])
vocab.set_default_index(vocab["<UNK>"])

target_pipeline = lambda x: x
text_pipeline = lambda x: vocab(x.split())


# batch split
def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    X = [vocab(text.split()) for text in X]
    max_words = max([len(i) for i in X])
    X = [tokens + ([0] * (max_words - len(tokens))) if len(tokens) < max_words else tokens[:max_words] for tokens in X]
    return torch.tensor(X), torch.tensor(Y)


dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, collate_fn=vectorize_batch)
testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=vectorize_batch)

# model hyper-parameters
if PRE_TRAIN:
    embed_len = embs_npa.shape[1]
else:
    embed_len = 128
hidden_dim = 128
n_layers = 3


'''
Build Model
'''


# Glove embedding
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


# RNN
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        if PRE_TRAIN:
            self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())
        else:
            self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len, bidirectional=True, hidden_size=hidden_dim, num_layers=n_layers,
                          batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        prev, hidden = self.rnn(embeddings, torch.randn(n_layers * 2, len(X_batch), hidden_dim))
        output = self.linear(prev)
        return output[:, -1]


# LSTM
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        if PRE_TRAIN:
            self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())
        else:
            self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.lstm = nn.LSTM(input_size=embed_len, bidirectional=True, hidden_size=hidden_dim, num_layers=n_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        hx = torch.randn(2 * n_layers, len(X_batch), hidden_dim)
        cx = torch.randn(2 * n_layers, len(X_batch), hidden_dim)
        prev, hidden = self.lstm(embeddings, (hx, cx))
        output = self.linear(prev)
        return output[:, -1]


# GRU
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        if PRE_TRAIN:
            self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())
        else:
            self.embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_len)
        self.GRU = nn.GRU(input_size=embed_len, bidirectional=True, hidden_size=hidden_dim, num_layers=n_layers,
                          batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        hx = torch.randn(2 * n_layers, len(X_batch), hidden_dim)
        cx = torch.randn(2 * n_layers, len(X_batch), hidden_dim)
        prev, hidden = self.GRU(embeddings, hx)
        output = self.linear(prev)
        return output[:, -1]


def CalcValLossAndAccuracy(model, loss_fn, val_loader):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [], [], []
        for X, Y in val_loader:
            preds = model(X).view(-1)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())

        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))


def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs):
    count = 0
    for i in range(1, epochs + 1):
        losses = []
        for X, Y in tqdm(train_loader, colour='white'):
            count += 1
            Y_preds = model(X).view(-1)

            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if count % 500 == 0:
                print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))

            if count % 1000 == 0:
                CalcValLossAndAccuracy(model, loss_fn, val_loader)


'''
Model train
'''
if MODEL == 'RNN':
    model_predictor = RNN()
elif MODEL == 'LSTM':
    model_predictor = LSTM()
elif MODEL == 'GRU':
    model_predictor = GRU()

# model save path
if PRE_TRAIN:
    PATH = 'model/'+MODEL + '_' + embed_model + '.pt'
else:
    PATH = 'model/'+MODEL + '.pt'

learning_rate = 1e-3
epoch = 1
loss_fn = nn.BCEWithLogitsLoss()

# start training
optimizer = Adam(model_predictor.parameters(), lr=learning_rate)
TrainModel(model_predictor, loss_fn, optimizer, dataloader, testloader, epoch)
# save model
torch.save(model_predictor, PATH)

# Load
model = torch.load(PATH)
model.eval()


def vectorize_val(batch):
    Y, X = list(zip(*batch))
    X = [vocab(text.split()) for text in X]
    max_words = max([len(i) for i in X])
    X = [tokens + ([0] * (max_words - len(tokens))) if len(tokens) < max_words else tokens[:max_words] for tokens in X]

    Y = [vocab(text.split()) for text in Y]
    max_words = max([len(i) for i in Y])
    Y = [tokens + ([0] * (max_words - len(tokens))) if len(tokens) < max_words else tokens[:max_words] for tokens in Y]
    return torch.tensor(X), torch.tensor(Y)


Test = pd.read_csv('new_val.csv')
Test = Test[['less_toxic', 'more_toxic']]
Test = Test.dropna(subset=['less_toxic', 'more_toxic'])
test_dataset = np.array(Test)

test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=vectorize_val)

preds = []
for X, Y in tqdm(test_loader, colour='white'):
    p1 = torch.sigmoid(model(X).view(-1)).detach().item()
    p2 = torch.sigmoid(model(Y).view(-1)).detach().item()
    if p1 > p2:
        preds.append(0)
    else:
        preds.append(1)
print('The average annotation score is:', 1 - sum(preds) / len(preds))
