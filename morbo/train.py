import numpy as np
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from datetime import date
from datetime import datetime
from early_stopping import EarlyStopping

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, optim

from rdkit import Chem
from rdkit.Chem import Draw
import selfies as sf
from chemVAE import main


def train(params,train_data,test_data):
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training Model on: " + str(device))

    # load data
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=True)
    save_path = ".\\"
    # initialize model
    model = main.VAE(params).to(device)
    early_stopping = EarlyStopping(save_path)
    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # set KL annealing
    KLD_alpha = np.linspace(0, 1, params['epochs'])

    ## generate unique filenames
    date_time = date.today()
    now = datetime.now()
    time = now.strftime("%H%M%S")
    model_filename = "model_" + str(date_time)

    # train model
    epoch = params['epochs']
    train_loss = []
    test_loss = []
    BCE_loss = []
    KLD_loss = []
    KLD_weight = []
    for epoch in range(1, epoch + 1):
        alpha = KLD_alpha[epoch - 1]
        loss, BCE, KLD_wt, KLD = main.train(model, train_loader, optimizer, device, epoch, alpha)
        train_loss.append(loss)
        BCE_loss.append(BCE)
        KLD_loss.append(KLD)
        KLD_weight.append(KLD_wt)

        test_loss.append(main.test(model, test_loader, optimizer, device, epoch, alpha))

        # save model
    ## save model paramters
    output = open(model_filename + '_parameters.pkl', 'wb')
    pickle.dump(params, output)
    output.close()
    print("Saved PyTorch Parameters State to " + model_filename + '_parameters.pkl')

    ## save model state
    torch.save(model.state_dict(), model_filename + '_state_epoch300.pth')
    print("Saved PyTorch Model State to " + model_filename + '_state.pth')

    ## save model state
    torch.save(model, model_filename + '.pth')
    print("Saved PyTorch Model State to " + model_filename + '_state.pth')




if __name__ == '__main__':
    # load dataset
    df = pd.read_csv('/home/leo/fightinglee/Antibiotic-project/data/test_train_data/chemble_dataset.csv')

    # select SMILEs data
    smiles = df['SMILES'].values
    selfies = main.smiles2selfies(smiles)
    onehot_selfies, idx_to_symbol = main.onehotSELFIES(selfies)

    # split data in training and testing
    X_train, X_test, y_train, y_test = train_test_split(onehot_selfies, onehot_selfies, test_size=0.40)
    X_test, X_val, y_test, y_val = train_test_split(X_test, X_test, test_size=0.50)

    # Pytroch Dataset
    train_data = main.SELFIES_Dataset(X_train, y_train, transform=transforms.ToTensor())
    test_data = main.SELFIES_Dataset(X_test, y_test, transform=transforms.ToTensor())
    val_data = main.SELFIES_Dataset(X_val, y_val, transform=transforms.ToTensor())

    num_characters, max_seq_len = onehot_selfies[0].shape

    params = {'num_characters': num_characters,
              'seq_length': max_seq_len,
              'num_conv_layers': 3,
              'layer1_filters': 24,
              'layer2_filters': 24,
              'layer3_filters': 24,
              'layer4_filters': 24,
              'kernel1_size': 11,
              'kernel2_size': 11,
              'kernel3_size': 11,
              'kernel4_size': 11,
              'lstm_stack_size': 3,
              'lstm_num_neurons': 396,
              'latent_dimensions': 256,
              'batch_size': 1024,
              'epochs': 300,
              'learning_rate': 10 ** -4}
    train(params,train_data,test_data)