# coding:utf-8
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


tr_path = 'train.csv'
tt_path = 'testA.csv'


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'



class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode

        df = pd.read_csv(path)
        df.drop(columns='id', inplace=True)
        df = df.join(df.heartbeat_signals.str.split(',', expand=True).astype('float'))
        df.drop(columns='heartbeat_signals', inplace=True)

        if not target_only:
            feats = list(range(93))
        else:
            pass

        if mode == 'test':
            data = df.values
            self.data = torch.FloatTensor(data)
        else:
            target = df['label'].values
            data = df.drop(columns='label').values

            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]

            self.data = torch.FloatTensor(data[indices])
            self.target = torch.LongTensor(target[indices])
            # self.target = torch.nn.functional.one_hot(self.target)

        self.dim = self.data.shape[1]

        print(
            f'Finished reading the {mode} set of COVID19 Dataset ({len(self.data)} samples found, each dim = {self.dim})')

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'), drop_last=False, num_workers=n_jobs,
                            pin_memory=True)
    return dataloader


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)


def test(tt_set, model, device):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            pred = torch.softmax(pred, dim=1)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

config = {
    'n_epochs': 3000,
    'batch_size': 256,
    'optimizer': 'SGD',
    'optim_hparas': {
        'lr': 0.01,
        'momentum': 0.9,
    },
    'early_stop': 200,
    'save_path': 'models/model.pth',
}

device = get_device()
os.makedirs('models', exist_ok=True)
target_only = False

tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(tt_set.dataset.dim).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = test(tt_set, model, device)
print(preds)
df = pd.read_csv('sample_submit.csv')
df[['label_0', 'label_1', 'label_2', 'label_3']] = preds
df.to_csv('submission_torch.csv', index=False)