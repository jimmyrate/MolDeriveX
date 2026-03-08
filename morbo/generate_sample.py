import numpy as np
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import logging
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

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger("ROOT")

# def generate_sample(dataset,model,idx_to_symbol):
#     # grab random sample from test_data
#     sample_idx = np.random.randint(0, len(dataset) - 1)
#     img, label = dataset[sample_idx]
#
#     # run model
#     with torch.no_grad():
#         img = img.to(device)
#         recon_data, z, mu, logvar = model(img)
#     recon_data = recon_data[0].cpu() #recon_data.shape =torch.Size([batch_size, vocab_size, max_len])
#
#     # grab original smiles
#     sample = img[0].cpu().numpy()
#     char_ind = list(np.argmax(sample.squeeze(), axis=0))
#     string = [idx_to_symbol[i] for i in char_ind]
#     selfie = ''.join(string)
#     smiles = sf.decoder(selfie)
#
#     # reconstructed smiles
#     recon_sample = recon_data.numpy()
#     char_ind = list(np.argmax(recon_sample.squeeze(), axis=0))
#     string = [idx_to_symbol[i] for i in char_ind]
#     recon_selfie = ''.join(string)
#     recon_smiles = sf.decoder(recon_selfie)
#     ## draw molecule and reconstruction
#     m1 = Chem.MolFromSmiles(smiles)
#     Draw.MolToFile(m1, 'original.png')
#
#     m2 = Chem.MolFromSmiles(recon_smiles)
#     Draw.MolToFile(m2, 'reconstruct.png')
#
#     ## visualize molecules in notebokk
#     figure(figsize=(8, 6), dpi=100)
#     plt.subplot(1, 2, 1)
#     img = mpimg.imread('original.png')
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title('Original')
#
#     plt.subplot(1, 2, 2)
#     img = mpimg.imread('reconstruct.png')
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title('Reconstruction')
#     plt.show()


def output2smiles(recon_data, idx_to_symbol):
    smiles_list = []
    for recon in recon_data:
        recon = recon.cpu()
        recon_sample = recon.numpy()  # Convert to numpy array
        char_ind = list(np.argmax(recon_sample.squeeze(), axis=0))  # Get indices of max values
        string = [idx_to_symbol[i] for i in char_ind]  # Convert indices to characters
        recon_selfie = ''.join(string)  # Join characters to form a SELFIES string
        recon_smiles = sf.decoder(recon_selfie)  # Decode SELFIES to SMILES
        smiles_list.append(recon_smiles)
    return smiles_list

def batch_generate_and_validate(dataset, model, idx_to_symbol, batch_size=32):
    # Device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # To store results
    valid_molecules = 0
    duplicates = 0
    training_smiles = set()  # Assuming you have a way to fill this with training set SMILES

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.eval()
    output_smi = []
    input_smi = []
    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            recon_data, z, mu, logvar = model(batch)

        # Convert output to SMILES
        batch_smiles = output2smiles(recon_data, idx_to_symbol)
        output_smi.extend(batch_smiles)


        batch_smiles_input = output2smiles(batch,idx_to_symbol)
        input_smi.extend(batch_smiles_input)

    output_mol = list(map(Chem.MolFromSmiles, output_smi))
    output_mol_valid = list(filter(None, output_mol))
    fraction_valid = len(output_mol_valid) / len(output_mol)

    logger.info(f"Valid molecules: {len(output_mol_valid)}/{len(output_mol)} ({fraction_valid * 100:.2f} %)")

    for output in output_smi:
        if output in input_smi:
            duplicates+=1
    print(duplicates)

def get_hidden(dataset,model,batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.eval()
    z_all = []
    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            z, mu, logvar = model.encoder(batch)
            z_all.append(z.cpu())
    z_all = torch.cat(z_all, dim=0)

    return z_all

def decoder_smiles(hidden,model,idx_to_symbol):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    output_smi = []
    with torch.no_grad():
        hidden = hidden.to(device)
        recon_data = model.decoder(hidden)
    output_smiles = output2smiles(recon_data,idx_to_symbol)
    output_smi.extend(output_smiles)

    return output_smi



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #pkl_path = '/home/leo/fightinglee/Antibiotic-project/chemVAE-main/chemVAE/model_2023-12-11_parameters_epoch300.pkl'
    pkl_path = '/root/morbo/new_vae/chemVAE/model_2023-12-11_parameters_epoch300.pkl'
    ## load parameters
    pkl_file = open(pkl_path, 'rb')
    params = pickle.load(pkl_file)
    pkl_file.close()

    ## load model state
    model = main.VAE(params).to(device)
    #model_state_path = '/home/leo/fightinglee/Antibiotic-project/chemVAE-main/chemVAE/model_2023-12-11_state_epoch300.pth'
    model_state_path = '/root/morbo/new_vae/chemVAE/model_2023-12-11_state_epoch300.pth'
    model.load_state_dict(torch.load(model_state_path))

    #df = pd.read_csv('/home/leo/fightinglee/Antibiotic-project/data/postive_data_new_for_train_clean.csv')
    df = pd.read_csv('/root/morbo/new_vae/postive_data_new_for_train_clean.csv')
    
    smiles = df['SMILES'].values
    selfies = main.smiles2selfies(smiles)
    onehot_selfies, idx_to_symbol = main.onehotSELFIES(selfies)

    # test_data = main.SELFIES_Dataset(onehot_selfies, onehot_selfies, transform=transforms.ToTensor())

    # generate_sample(test_data,model,idx_to_symbol)
    # batch_generate_and_validate(onehot_selfies,model,idx_to_symbol,batch_size=1024)
    hidden = get_hidden(onehot_selfies,model,batch_size=1024)

    out_smiles = decoder_smiles(hidden,model,idx_to_symbol)
    print(out_smiles[0])

