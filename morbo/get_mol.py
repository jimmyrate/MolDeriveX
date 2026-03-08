from typing import Optional
from botorch.test_functions.base import MultiObjectiveTestProblem
import torch
from moses.metrics import QED as QED_
from moses.metrics import SA, logP
import numpy as np
from rdkit import Chem
import pickle
from rdkit.Chem import AllChem
from rdkit import DataStructs
from problems.computer_mol_similarity import calculate_active, calculate_tanimoto
import csv
from torch.utils.data import DataLoader
import selfies as sf
import logging
#from chemVAE import main
from lolbo.molecule_objective import MoleculeObjective
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger("ROOT")
# from guacamol import standard_benchmarks

# med1 = standard_benchmarks.median_camphor_menthol() #'Median molecules 1'
# med2 = standard_benchmarks.median_tadalafil_sildenafil() #'Median molecules 2',
# pdop = standard_benchmarks.perindopril_rings() # 'Perindopril MPO',
# osmb = standard_benchmarks.hard_osimertinib()  # 'Osimertinib MPO',
# adip = standard_benchmarks.amlodipine_rings()  # 'Amlodipine MPO' 
# siga = standard_benchmarks.sitagliptin_replacement() #'Sitagliptin MPO'
# zale = standard_benchmarks.zaleplon_with_other_formula() # 'Zaleplon MPO'
# valt = standard_benchmarks.valsartan_smarts()  #'Valsartan SMARTS',
# dhop = standard_benchmarks.decoration_hop() # 'Deco Hop'
# shop = standard_benchmarks.scaffold_hop() # Scaffold Hop'
# rano= standard_benchmarks.ranolazine_mpo() #'Ranolazine MPO' 
# fexo = standard_benchmarks.hard_fexofenadine() # 'Fexofenadine MPO'... 'make fexofenadine less greasy'

def QED(mol):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return -(QED_(mol))
    except:
        return 0
    
# def penalized_logP(mol):
#     """Penalized logP.

#     Computed as logP(mol) - SA(mol) as in JT-VAE.

#     Args:
#         mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

#     Returns:
#         float: Penalized logP or NaN if mol is None.
#     """
#     try:
#         value = logP(mol) - SA(mol)
#         if value < -8:
#             value = -8
#         elif value > 32:
#             value = 32
#         value  = (value+8)/40
#         return -value
#     except:
#         return 0
def get_SA(mol):
    try:
        value = SA(mol)
        value = (value - 1)/9
        return value-1
    except:
        return 0
class jnk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = '/root/morbo/experiments/data/jnk3/jnk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = jnk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)
def get_jnk(smiles):
    jnk = jnk3_model()(smiles)
    return -jnk[0]
class gsk3_model():
    """Scores based on an ECFP classifier for activity."""

    kwargs = ["clf_path"]
    clf_path = '/root/morbo/morbo/problems/data/gsk3/gsk3.pkl'

    def __init__(self):
        with open(self.clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = gsk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)
def get_gsk3(smiles):
    gsk = gsk3_model()(smiles)
    return -gsk[0]

def get_sim(gen_molecule):

    if gen_molecule == None:
        return 0
    
    path = "/root/morbo/morbo/train-negative_clean_token.csv"
    training_data = []
    with open(path) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            training_data.append(row[0])
    training_data = training_data[1:]
    #gen_molecule = gen_molecule[0]
    similarity = max(calculate_tanimoto(gen_molecule, train_molecule) for train_molecule in training_data)
    return -similarity

def get_active(gen_molecule):
    #gen_molecule = gen_molecule[0]
    if gen_molecule == None:
        return 0
    active = calculate_active(gen_molecule)
    return -active

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
def get_mol_fn(dim, model, idx_to_symbol,device= None, dtype = None):

    tkwargs = {"device": device, "dtype": dtype}
    lb, ub = -5*torch.ones(dim, **tkwargs).to('cuda'), 5*torch.ones(dim, **tkwargs).to('cuda')
    bounds = torch.stack((lb, ub), dim=0)

    def f(x):
        n = len(x)
        objs= []
        for i in range(n):
            z = torch.reshape(x[i],(1,-1)).to(torch.float32)
            dec_smiles=decoder_smiles(z,model,idx_to_symbol)
            #_ ,dec_smiles = model.vae_decode(z)
            # try:
            #     _ ,dec_smiles = model.vae_decode(z)
            # except:
            #     print(z)
            #     print(i)
            #dec_mol = Chem.MolFromSmiles(dec_smiles[0])
            #pen_logP = penalized_logP(dec_mol)
            #mol_rano = -rano.objective.score(dec_smiles[0])
            #mol_pdop = -pdop.objective.score(dec_smiles[0])
            #mol_zale = -zale.objective.score(dec_smiles[0])
            #mol_QED = QED(dec_mol)
            #mol_SA = get_SA(dec_mol)
            #mol_gsk = get_gsk3(dec_smiles)
            #mol_jnk = get_jnk(dec_smiles)
            dec_smiles = dec_smiles[0]
            mol_sim = get_sim(dec_smiles)
            mol_active = get_active(dec_smiles)
            objs.append([mol_sim,mol_active])
        objs = torch.tensor(objs).to('cuda')
        return objs
    return f, bounds

def decode_mol(x, model):
    
    n = len(x)
    smiles = []
    similarity = []
    activity = []
    for i in range(n):
        z = torch.reshape(x[i],(1,-1)).to(torch.float32)
        _ ,dec_smiles = model.vae_decode(z)
        dec_smiles = dec_smiles[0]
        mol_sim = -get_sim(dec_smiles)
        mol_active = -get_active(dec_smiles)

        smiles.append(dec_smiles)
        similarity.append(mol_sim)
        activity.append(mol_active)

    data = {"smiles":smiles,"similarity":similarity,"activity":activity}
    df = pd.DataFrame(data)
    df.to_csv("/root/morbo/morbo/decoder_mol_seed_03.csv",index = False)

def return_mol(x,model):
    n = len(x)
    for i in range(n):
        z = torch.reshape(x[i],(1,-1)).to(torch.float32)
        _ ,dec_smiles = model.vae_decode(z)
        dec_smiles = dec_smiles[0]
        print(dec_smiles)

if __name__ == "__main__":
    
    # df = pd.read_csv('/root/morbo/morbo/test_data.csv')
    # smiles = df['SMILES'].values
    # selfies = main.smiles2selfies(smiles)
    # onehot_selfies, idx_to_symbol = main.onehotSELFIES(selfies)

    # pkl_path = '/root/morbo/new_vae/chemVAE/model_2023-12-11_parameters_epoch300.pkl'
    # ## load parameters
    # pkl_file = open(pkl_path, 'rb')
    # params = pickle.load(pkl_file)
    # pkl_file.close()

    # ## load model state
    # model = main.VAE(params).to('cuda')
    # #model_state_path = '/home/leo/fightinglee/Antibiotic-project/chemVAE-main/chemVAE/model_2023-12-11_state_epoch300.pth'
    # model_state_path = '/root/morbo/new_vae/chemVAE/model_2023-12-11_state_epoch300.pth'
    # model.load_state_dict(torch.load(model_state_path))
    # pkl_path = '/root/morbo/morbo/chemVAE/model_2024-01-08_parameters_epoch100.pkl'
    # ## load parameters
    # pkl_file = open(pkl_path, 'rb')
    # params = pickle.load(pkl_file)
    # pkl_file.close()

    # ## load model state
    # model = main.VAE(params).to('cuda')
    # #model_state_path = '/home/leo/fightinglee/Antibiotic-project/chemVAE-main/chemVAE/model_2023-12-11_state_epoch300.pth'
    # model_state_path = '/root/morbo/morbo/chemVAE/model_2024-01-08_state_epoch100.pth'
    # model.load_state_dict(torch.load(model_state_path))

    # df = pd.read_csv('/root/morbo/morbo/TRAIN_VAE_dataset.csv')
    # smiles = df['SMILES'].values
    # selfies = main.smiles2selfies(smiles)
    # onehot_selfies, idx_to_symbol = main.onehotSELFIES(selfies)
    BASE_SEED = 12346
    seed = BASE_SEED + 3
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MoleculeObjective(task_id = 'logp')
    names = ['/root/morbo/experiments/mol_selfies/morbo/0003_morbo_10001.pt']
    # names = ['/root/morbo/experiments/mol_selfies/morbo/0001_morbo_10000.pt',
    #         '/root/morbo/experiments/mol_selfies/morbo/0002_morbo_10000.pt']
    datas = []

    for name in names:
        data = torch.load(name)
        x = data['true_pareto_X'][-1]
        x = torch.tensor(x)
        datas.append(x)
    datas = torch.cat(datas)
    #print(datas)
    # #return_mol(datas,model)
    decode_mol(datas,model)
    # df = pd.read_csv("/root/morbo/morbo/train-negative_clean.csv")
    # all_smiles = df['SMILES'].values
    
    # sim_values = []
    # active_values = []
    # for smiles in all_smiles:
        
    #     sim_value = get_sim(smiles)
    #     active_value = get_active(smiles)
        
    #     sim_values.append(sim_value)
    #     active_values.append(active_value)
    # data = {'sim':sim_values, 'act':active_values}
    # df = pd.DataFrame(data)
    # df.to_csv('/root/morbo/morbo/positive_mol_value.csv',index = False)
    #     #print(f"sim:{sim_value}, act:{active_value}")
    # BASE_SEED = 12346
    # seed = BASE_SEED + 2
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # names = ['/root/morbo/experiments/mol_selfies/morbo/0002_morbo_10000.pt']
    # model = MoleculeObjective(task_id = 'logp')
    # data = torch.load(names[0])
    # x = data['true_pareto_X'][-1][0]
    # x = torch.tensor(x)
    # #print(x.shape)
    # z = torch.reshape(x,(1,-1)).to(torch.float32)
    # _ ,dec_smiles = model.vae_decode(z)
    # dec_smiles = dec_smiles[0]
    # print(dec_smiles)
    # x1 = x.clone()
    # z1 = torch.reshape(x1,(1,-1)).to(torch.float32)
    # _ ,dec_smiles = model.vae_decode(z1)
    # dec_smiles = dec_smiles[0]
    # print(dec_smiles)
