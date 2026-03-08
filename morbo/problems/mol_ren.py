from typing import Optional
from botorch.test_functions.base import MultiObjectiveTestProblem
from tqdm import tqdm
import torch
import torch.nn as nn
from moses.metrics import QED as QED_
from moses.metrics import SA, logP
import numpy as np
from rdkit import Chem
import pickle
from rdkit.Chem import AllChem
from rdkit import DataStructs
# from .computer_mol_similarity_chemprop import calculate_active, calculate_tanimoto,calculate_chemprop_active
import csv
from .computer_mol_similarity import calculate_tanimoto,calculate_active
from torch.utils.data import DataLoader
import selfies as sf
import logging
from collections import OrderedDict
from typing import List, Optional, Union, Tuple
from chemprop.train.predict import predict
from chemprop.spectra_utils import normalize_spectra, roundrobin_sid
from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit, update_prediction_args
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_reaction, set_explicit_h, reset_featurization_parameters
from chemprop.models import MoleculeModel
import random
import joblib
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import xgboost as xgb
# from corrector.src.invalidSMILES import get_invalid_smiles
# # from corrector.src.preprocess import standardization_pipeline, remove_smiles_duplicates
# from corrector.src.modelling import initialize_model, correct_SMILES
import pandas as pd

class objHelper:
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    def save_obj(self, obj):
        """save obj with pickle"""
        with open(self.file_name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self):
        """load a pickle object"""
        try:
            with open(self.file_name, 'rb') as f:
                return pickle.load(f)
        except:
            return None


class s2t(object):
    def __init__(self, filename):
        self.t2v = {}
        self.dim = None
        for line in open(filename):
            line = line.strip('\n')
            line = line.split('\t')
            t = line[0]
            v = np.array([float(x) for x in line[1].split()])
            if self.dim is None:
                self.dim = len(v)
            else:
                v = v[:self.dim]
            self.t2v[t] = v

    def embed(self, seq):
        if seq.find(' ') > 0:
            s = seq.strip().split()
        else:
            s = list(seq.strip())

        rst = []
        for x in s:
            v = self.t2v.get(x)
            if v is None:
                continue
            rst.append(v)
        return np.array(rst)

    def embed_normalized(self, seq, length=60):
        rst = self.embed(seq)
        if len(rst) > length:
            return rst[:length]
        elif len(rst) < length:
            return np.concatenate((rst, np.zeros((length - len(rst), self.dim))))
        return rst

    def embed_RECM_position(self, seq):
        if seq.find(' ') > 0:
            s = seq.strip().split()
        else:
            s = list(seq.strip())
        order_char = ['A', 'C', 'D', 'E',
                      'F', 'G', 'H', 'I',
                      'K', 'L', 'M', 'N',
                      'P', 'Q', 'R', 'S',
                      'T', 'V', 'W', 'Y']
        rst = []
        for x in order_char:
            x_repeat = s.count(x)
            if x_repeat == 0:
                v = np.zeros(20)
            else:
                v = self.t2v.get(x)
                v = v * x_repeat
            v = v.tolist()
            rst = rst + v
        # rst = np.array(rst)

        # rst = rst.flatten()
        return np.array(rst)



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

def seq_to_feature(seq):

    seq2t = s2t('/root/morbo/TransVAE-master/AFP-svm/features/RECM-position.txt')
    seq_array = seq2t.embed_RECM_position(seq)
    return seq_array

def get_sim(gen_molecule):

    if gen_molecule == None:
        return 0
    
    path = "/root/morbo/TransVAE-master/data/antibiotic/FDA_Gram-Positive_Bacteria_Antibiotics.csv"
    training_data = []
    with open(path) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            training_data.append(row[0])
    training_data = training_data[1:]
    #gen_molecule = gen_molecule[0]
    similarity = max(calculate_tanimoto(gen_molecule, train_molecule) for train_molecule in training_data)
    return -similarity

# def get_active(gen_molecule):
#     #gen_molecule = gen_molecule[0]
#     if gen_molecule == None:
#         return 0
#
#     active = calculate_chemprop_active(args=PredictArgs().parse_args(),gen_molecule=[gen_molecule])
#     # active = calculate_active(gen_molecule)
#     return -active
# def corrector_smiles(smiles):
#     #model parameters
#     folder_out = "/root/morbo/corrector/Data/"
#     initialize_source = '/root/morbo/corrector/Data/papyrus_rnn_XS.csv'
#     invalid_type = 'multiple'
#     num_errors = 12
#     threshold = 200
#     data_source = f"PAPYRUS_{threshold}"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model, out, SRC = initialize_model(folder_out,
#                                        data_source,
#                                        error_source=initialize_source,
#                                        device=device,
#                                        threshold=threshold,
#                                        epochs=30,
#                                        layers=3,
#                                        batch_size=16,
#                                        invalid_type=invalid_type,
#                                        num_errors=num_errors)
#     valids1, df_output1 = correct_SMILES(model, out, smiles, device, SRC)
#     out = []
#     for index, row in df_output1.iterrows():
#         if not pd.isnull(row['CORRECT']) and row['CORRECT'] != 'nan':
#             out.append(row['CORRECT'])
#         elif not pd.isnull(row['ORIGINAL']) and row['ORIGINAL'] != 'nan':
#             out.append(row['ORIGINAL'])
#     out = list(filter(None, out))
#     return out

def get_active(gen_molecule):
    #gen_molecule = gen_molecule[0]
    if gen_molecule == None:
        return 0
    args = PredictArgs().parse_args()
    args.checkpoint_dir = '/root/morbo/chemprop/abaucin_models/final_model'
    args.checkpoint_paths = ['/root/morbo/chemprop/abaucin_models/final_model/fold_0/model_0/model.pt',
                             # '/root/morbo/chemprop/antibiotic_regression_checkpoints/fold_1/model_0/model.pt',
                             # '/root/morbo/chemprop/antibiotic_regression_checkpoints/fold_2/model_0/model.pt',
                             # '/root/morbo/chemprop/antibiotic_regression_checkpoints/fold_3/model_0/model.pt',
                             # '/root/morbo/chemprop/antibiotic_regression_checkpoints/fold_4/model_0/model.pt',
                             ]
    active = calculate_chemprop_active(args=args,gen_molecule=[gen_molecule])

    # # 确保最终值在0-1之间
    # active = max(0, min(1, active))
    # active = 1 - active

    # active = calculate_active(gen_molecule)
    return -active

def get_chemprop_result(gen_molecule):
    #gen_molecule = gen_molecule[0]
    if gen_molecule == None:
        return 0
    active_all = []
    args = PredictArgs().parse_args()
    args.checkpoint_dir = '/root/morbo/chemprop/antibiotic_regression(MIC)_checkpoints'
    args.checkpoint_paths = ['/root/morbo/chemprop/antibiotic_regression(MIC)_checkpoints/fold_0/model_0/model.pt',
                             '/root/morbo/chemprop/antibiotic_regression(MIC)_checkpoints/fold_1/model_0/model.pt',
                             '/root/morbo/chemprop/antibiotic_regression(MIC)_checkpoints/fold_2/model_0/model.pt',
                             '/root/morbo/chemprop/antibiotic_regression(MIC)_checkpoints/fold_3/model_0/model.pt',
                             '/root/morbo/chemprop/antibiotic_regression(MIC)_checkpoints/fold_4/model_0/model.pt',
                             ]

    active = calculate_chemprop_active(args=args,gen_molecule=[gen_molecule])
    active_all.append(active)
    return active_all

def get_chemberta_result(gen_molecule):
    # 定义五折模型的路径
    model_paths = [
        "/root/morbo/data_classification/logMIC_model_fold_1/checkpoint",
        "/root/morbo/data_classification/logMIC_model_fold_2/checkpoint",
        "/root/morbo/data_classification/logMIC_model_fold_3/checkpoint",
        "/root/morbo/data_classification/logMIC_model_fold_4/checkpoint",
        "/root/morbo/data_classification/logMIC_model_fold_5/checkpoint"
    ]

    # 用来存储每个 SMILES 的五折结果
    all_results = []

    # 对于列表中的每个 SMILES 进行处理

        # 用来存储当前 SMILES 的每个折模型的结果
    fold_results = []

    for model_path in model_paths:
            # 加载模型和tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained('/root/morbo/ChemBERTa-77M-MTR')

            # 对当前 SMILES 进行tokenize
        tokenized_samples = tokenizer(gen_molecule, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

            # 禁用梯度计算，因为我们只进行推理
        with torch.no_grad():
                # 获取模型输出
            outputs = model(**tokenized_samples)
            logits = outputs.logits

                # 计算概率，假设输出是回归问题（logMIC 值为连续值预测）
            probabilities = logits.squeeze().item()

            # 将当前折模型的预测结果存入列表
        fold_results.append(probabilities)

        # 计算当前 SMILES 的五折结果的均值
    final_result = sum(fold_results) / len(fold_results)

        # 将最终结果存入所有 SMILES 的结果列表中
    all_results.append(final_result)

    return all_results

def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

def get_xgboost_result(gen_molecule):
    # 定义5个模型路径
    model_paths = [
        "/root/morbo/data_classification/xgboost_model/xgboost_model_fold_1.joblib",
        "/root/morbo/data_classification/xgboost_model/xgboost_model_fold_2.joblib",
        "/root/morbo/data_classification/xgboost_model/xgboost_model_fold_3.joblib",
        "/root/morbo/data_classification/xgboost_model/xgboost_model_fold_4.joblib",
        "/root/morbo/data_classification/xgboost_model/xgboost_model_fold_5.joblib"
    ]

    all_results = []  # 存储每个分子的最终结果


        # 将 SMILES 转换为分子指纹
    fingerprint = smiles_to_fingerprint(gen_molecule)
    if fingerprint is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

        # 存储当前分子的每个模型预测结果
    fold_results = []

    for model_path in model_paths:
            # 加载训练好的 XGBoost 模型
        model = joblib.load(model_path)

            # 对当前模型进行预测
        prediction = model.predict(fingerprint.reshape(1, -1))  # 重塑数据以适应模型输入格式

            # 存储每个模型的预测结果
        fold_results.append(float(prediction))

        # 对当前分子的5折预测结果取均值
    final_result = sum(fold_results) / len(fold_results)
    all_results.append(final_result)

    return all_results

def use_stacking_model(gen_molecule):
    # 定义 5 折 Stacking 模型的路径
    model_paths = [
        "/root/morbo/data_classification/stacking_model/stacking_model_fold_1.model",
        "/root/morbo/data_classification/stacking_model/stacking_model_fold_2.model",
        "/root/morbo/data_classification/stacking_model/stacking_model_fold_3.model",
        "/root/morbo/data_classification/stacking_model/stacking_model_fold_4.model",
        "/root/morbo/data_classification/stacking_model/stacking_model_fold_5.model"
    ]

    # 获取基模型的预测结果
    fingerprint_preds = get_xgboost_result(gen_molecule)
    graph_preds = get_chemprop_result(gen_molecule)
    sequence_preds = get_chemberta_result(gen_molecule)


    # 检查所有模型输出长度是否一致
    if not (len(graph_preds) == len(sequence_preds) == len(fingerprint_preds)):
        raise ValueError("基模型的预测结果长度不一致，请检查输入数据")

    # 将所有基模型的预测结果拼接成一个 (N, 3) 的矩阵，N 是分子数量
    stacked_features = np.column_stack([graph_preds, sequence_preds, fingerprint_preds])

    # 将拼接后的特征转换为 XGBoost 的 DMatrix 类型
    dmatrix_stacked_features = xgb.DMatrix(stacked_features)

    # 存储每折 Stacking 模型的预测结果
    stacking_results = []

    # 对每个 Stacking 模型进行预测
    for model_path in model_paths:
        # 加载训练好的 Stacking 模型
        stacking_model = joblib.load(model_path)

        # 使用 Stacking 模型进行预测
        fold_preds = stacking_model.predict(dmatrix_stacked_features)

        # 存储每个折的预测结果
        stacking_results.append(fold_preds)

    # 将 5 折 Stacking 模型的预测结果取均值
    final_result = np.mean(stacking_results, axis=0)

    final_result = max(0, min(1, final_result))

    final_result = 1 - final_result

    return -final_result



def get_active_EP2(gen_molecule):
    #gen_molecule = gen_molecule[0]
    if gen_molecule == None:
        return 0
    args_ep2 = PredictArgs().parse_args()
    args_ep2.checkpoint_dir = '/root/morbo/chemprop/EP2_checkpoints'
    args_ep2.checkpoint_paths = ['/root/morbo/chemprop/EP2_checkpoints/fold_0/model_0/model.pt',
                                 '/root/morbo/chemprop/EP2_checkpoints/fold_1/model_0/model.pt',
                                 '/root/morbo/chemprop/EP2_checkpoints/fold_2/model_0/model.pt',
                                 '/root/morbo/chemprop/EP2_checkpoints/fold_3/model_0/model.pt',
                                 '/root/morbo/chemprop/EP2_checkpoints/fold_4/model_0/model.pt',
                                 '/root/morbo/chemprop/EP2_checkpoints/fold_5/model_0/model.pt',
                                 '/root/morbo/chemprop/EP2_checkpoints/fold_6/model_0/model.pt',
                                 '/root/morbo/chemprop/EP2_checkpoints/fold_7/model_0/model.pt',
                                 '/root/morbo/chemprop/EP2_checkpoints/fold_8/model_0/model.pt',
                                 '/root/morbo/chemprop/EP2_checkpoints/fold_9/model_0/model.pt']
    active_EP2 = calculate_chemprop_active(args=args_ep2,gen_molecule=[gen_molecule])
    # active = calculate_active(gen_molecule)
    return -active_EP2

def get_active_EP4(gen_molecule):
    #gen_molecule = gen_molecule[0]
    if gen_molecule == None:
        return 0
    args_ep4 = PredictArgs().parse_args()
    args_ep4.checkpoint_dir = '/root/morbo/chemprop/EP4_checkpoints'
    args_ep4.checkpoint_paths = ['/root/morbo/chemprop/EP4_checkpoints/fold_0/model_0/model.pt',
                                 '/root/morbo/chemprop/EP4_checkpoints/fold_1/model_0/model.pt',
                                 '/root/morbo/chemprop/EP4_checkpoints/fold_2/model_0/model.pt',
                                 '/root/morbo/chemprop/EP4_checkpoints/fold_3/model_0/model.pt',
                                 '/root/morbo/chemprop/EP4_checkpoints/fold_4/model_0/model.pt',
                                 '/root/morbo/chemprop/EP4_checkpoints/fold_5/model_0/model.pt',
                                 '/root/morbo/chemprop/EP4_checkpoints/fold_6/model_0/model.pt',
                                 '/root/morbo/chemprop/EP4_checkpoints/fold_7/model_0/model.pt',
                                 '/root/morbo/chemprop/EP4_checkpoints/fold_8/model_0/model.pt',
                                 '/root/morbo/chemprop/EP4_checkpoints/fold_9/model_0/model.pt']
    active_EP4 = calculate_chemprop_active(args=args_ep4,gen_molecule=[gen_molecule])
    # active = calculate_active(gen_molecule)
    return -active_EP4

def get_AFP_active(peptides):
    model_path = '/root/morbo/TransVAE-master/AFP-svm/model_pkl/svr_model_AFP_uniprot.pkl'
    svm = objHelper(model_path).load_obj()
    peptide_feature = seq_to_feature(peptides)
    prediction_proba = svm.predict_proba(peptide_feature .reshape(1, -1))
    prediction_score = [x[1] for x in prediction_proba]
    active = float(prediction_score[0])
    return -active

def output2smiles(recon_data, idx_to_symbol, nop_token='[nop]'):
    smiles_list = []
    for recon in recon_data:
        recon = recon.cpu()
        recon_sample = recon.numpy()  # Convert to numpy array
        char_ind = np.argmax(recon_sample.squeeze(), axis=0)  # Get indices of max values
        # Convert indices to characters, filter out the nop_token
        string = [idx_to_symbol[i] for i in char_ind if idx_to_symbol[i] != nop_token]
        recon_smile = ''.join(string)  # Join characters to form a SMILE string
        smiles_list.append(recon_smile)
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
    output_smiles = output2smiles(recon_data, idx_to_symbol)
    output_smi.extend(output_smiles)

    return output_smi

def get_mol_fn(dim, model,device= None, dtype = None):

    tkwargs = {"device": device, "dtype": dtype}
    lb, ub = -5.5*torch.ones(dim, **tkwargs).to('cuda'), 5*torch.ones(dim, **tkwargs).to('cuda')
    bounds = torch.stack((lb, ub), dim=0)
    def f(x):
        n = len(x)
        objs= []
        all_smiles = []
        for i in tqdm(range(n)):
            z = torch.reshape(x[i],(1,-1)).to(torch.float32)
            #dec_smiles=decoder_smiles(z,model,idx_to_symbol)
            #_ ,dec_smiles = model.vae_decode(z)
            dec_smiles = None
            try:
                # _ ,dec_smiles = model.vae_decode(z)
                dec_smiles = model.reconstruct_decoder(z)
                dec_smiles = dec_smiles[0]
                all_smiles.append(dec_smiles)
            except Exception as e:
                print("Exception occurred:", e)
                print(z)
            #dec_mol = Chem.MolFromSmiles(dec_smiles[0])
            #pen_logP = penalized_logP(dec_mol)
            #mol_rano = -rano.objective.score(dec_smiles[0])
            #mol_pdop = -pdop.objective.score(dec_smiles[0])
            #mol_zale = -zale.objective.score(dec_smiles[0])
            #mol_QED = QED(dec_mol)
            #mol_SA = get_SA(dec_mol)
            #mol_gsk = get_gsk3(dec_smiles)
            #mol_jnk = get_jnk(dec_smiles)

            if dec_smiles is not None and Chem.MolFromSmiles(dec_smiles):
                # print(dec_smiles)
                mol_sim = get_sim(dec_smiles)
                mol_active = get_active(dec_smiles)
                # mol_active = use_stacking_model(dec_smiles)
                #目标函数
                objs.append([mol_sim,mol_active])
            else:
                mol_sim = get_sim(None)
                mol_active = get_active(None)

                objs.append([mol_sim, mol_active])

        objs = torch.tensor(objs).to('cuda')
        return objs,all_smiles
    return f, bounds

def get_mol_fn_EP(dim, model,device= None, dtype = None):

    tkwargs = {"device": device, "dtype": dtype}
    lb, ub = -4.3*torch.ones(dim, **tkwargs).to('cuda'), 5.4*torch.ones(dim, **tkwargs).to('cuda')
    bounds = torch.stack((lb, ub), dim=0)

    def f(x):
        n = len(x)
        objs= []
        for i in tqdm(range(n)):
            z = torch.reshape(x[i],(1,-1)).to(torch.float32)
            #dec_smiles=decoder_smiles(z,model,idx_to_symbol)
            #_ ,dec_smiles = model.vae_decode(z)
            dec_smiles = None
            try:
                # _ ,dec_smiles = model.vae_decode(z)
                dec_smiles = model.reconstruct_decoder(z)
                dec_smiles = dec_smiles[0]
            except Exception as e:
                print("Exception occurred:", e)
                print(z)
            #dec_mol = Chem.MolFromSmiles(dec_smiles[0])
            #pen_logP = penalized_logP(dec_mol)
            #mol_rano = -rano.objective.score(dec_smiles[0])
            #mol_pdop = -pdop.objective.score(dec_smiles[0])
            #mol_zale = -zale.objective.score(dec_smiles[0])
            #mol_QED = QED(dec_mol)
            #mol_SA = get_SA(dec_mol)
            #mol_gsk = get_gsk3(dec_smiles)
            #mol_jnk = get_jnk(dec_smiles)

            if dec_smiles is not None and Chem.MolFromSmiles(dec_smiles):
                # print(dec_smiles)
                mol_active_EP2 = get_active_EP2(dec_smiles)
                mol_active_EP4 = get_active_EP4(dec_smiles)
                #目标函数
                objs.append([mol_active_EP2,mol_active_EP4])
            else:
                mol_active_EP2 = get_active_EP2(None)
                mol_active_EP4 = get_active_EP4(None)
                # 目标函数
                objs.append([mol_active_EP2, mol_active_EP4])

        objs = torch.tensor(objs).to('cuda')
        return objs
    return f, bounds

def get_mol_fn_peptides(dim, model, idx_to_symbol, device= None, dtype = None):

    tkwargs = {"device": device, "dtype": dtype}
    lb, ub = -5*torch.ones(dim, **tkwargs).to('cuda'), 5*torch.ones(dim, **tkwargs).to('cuda')
    bounds = torch.stack((lb, ub), dim=0)

    def f(x):
        n = len(x)
        objs= []
        for i in tqdm(range(n)):
            z = torch.reshape(x[i],(1,-1)).to(torch.float32)
            # dec_smiles=decoder_smiles(z,model,idx_to_symbol)
            # _,dec_smiles = model.vae_decode(z)
            dec_peptides = None
            try:
                dec_peptides = decoder_smiles(z, model, idx_to_symbol)
                # _ ,dec_peptides = model.vae_decode(z)
                # dec_peptides = model.reconstruct_decoder(z)
                dec_peptides = dec_peptides[0]
            except Exception as e:
                print("Exception occurred:", e)
                print(z)
            if dec_peptides is not None:
                # print(dec_smiles)
                peptides_toxic = calculate_peptide_toxic(dec_peptides)
                mol_active = get_AFP_active(dec_peptides)

                #目标函数
                objs.append([peptides_toxic,mol_active])
            else:
                peptides_toxic = None
                mol_active = None
                # 目标函数
                objs.append([peptides_toxic, mol_active])

        objs = torch.tensor(objs).to('cuda')
        return objs
    return f, bounds



def load_model(args: PredictArgs, generator: bool = False):
    """
    Function to load a model or ensemble of models from file. If generator is True, a generator of the respective model and scaler
    objects is returned (memory efficient), else the full list (holding all models in memory, necessary for preloading).

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param generator: A boolean to return a generator instead of a list of models and scalers.
    :return: A tuple of updated prediction arguments, training arguments, a list or generator object of models, a list or
                 generator object of scalers, the number of tasks and their respective names.
    """
    # print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[PredictArgs, TrainArgs]

    # Load model and scalers
    models = (load_checkpoint(checkpoint_path, device=args.device) for checkpoint_path in args.checkpoint_paths)
    scalers = (load_scalers(checkpoint_path) for checkpoint_path in args.checkpoint_paths)
    if not generator:
        models = list(models)
        scalers = list(scalers)

    return args, train_args, models, scalers, num_tasks, task_names

def load_data(args: PredictArgs, smiles: List[List[str]]):
    """
    Function to load data from a list of smiles or a file.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: A list of list of smiles, or None if data is to be read from file
    :return: A tuple of a :class:`~chemprop.data.MoleculeDataset` containing all datapoints, a :class:`~chemprop.data.MoleculeDataset` containing only valid datapoints,
                 a :class:`~chemprop.data.MoleculeDataLoader` and a dictionary mapping full to valid indices.
    """
    # print('Loading data')
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator
        )
    else:
        full_data = get_data(path=args.test_path, smiles_columns=args.smiles_columns, target_columns=[],
                             ignore_columns=[],
                             skip_invalid_smiles=False, args=args, store_row=not args.drop_extra_columns)

    # print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    # print(f'Test size = {len(test_data):,}')

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    return full_data, test_data, test_data_loader, full_to_valid_indices

def set_features(args: PredictArgs, train_args: TrainArgs):
    """
    Function to set extra options.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param train_args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    """
    reset_featurization_parameters()

    if args.atom_descriptors == 'feature':
        set_extra_atom_fdim(train_args.atom_features_size)

    if args.bond_features_path is not None:
        set_extra_bond_fdim(train_args.bond_features_size)

    # set explicit H option and reaction option
    set_explicit_h(train_args.explicit_h)
    set_reaction(train_args.reaction, train_args.reaction_mode)


def predict_and_save(args: PredictArgs, train_args: TrainArgs, test_data: MoleculeDataset,
                     task_names: List[str], num_tasks: int, test_data_loader: MoleculeDataLoader,
                     full_data: MoleculeDataset,
                     full_to_valid_indices: dict, models: List[MoleculeModel], scalers: List[List[StandardScaler]]):
    """
    Function to predict with a model and save the predictions to file.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param train_args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param test_data: A :class:`~chemprop.data.MoleculeDataset` containing valid datapoints.
    :param task_names: A list of task names.
    :param num_tasks: Number of tasks.
    :param test_data_loader: A :class:`~chemprop.data.MoleculeDataLoader` to load the test data.
    :param full_data:  A :class:`~chemprop.data.MoleculeDataset` containing all (valid and invalid) datapoints.
    :param full_to_valid_indices: A dictionary dictionary mapping full to valid indices.
    :param models: A list or generator object of :class:`~chemprop.models.MoleculeModel`\ s.
    :param scalers: A list or generator object of :class:`~chemprop.features.scaler.StandardScaler` objects.
    :return:  A list of lists of target predictions.
    """
    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), num_tasks))
    if args.ensemble_variance or args.individual_ensemble_predictions:
        if args.dataset_type == 'multiclass':
            all_preds = np.zeros((len(test_data), num_tasks, args.multiclass_num_classes, len(args.checkpoint_paths)))
        else:
            all_preds = np.zeros((len(test_data), num_tasks, len(args.checkpoint_paths)))

    # Partial results for variance robust calculation.
    # print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    # for index, (model, scaler_list) in enumerate(zip(models, scalers), total=len(args.checkpoint_paths)):
    for index, (model, scaler_list) in enumerate(zip(models, scalers)):
        scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = scaler_list

        # Normalize features
        if args.features_scaling or train_args.atom_descriptor_scaling or train_args.bond_feature_scaling:
            test_data.reset_features_and_targets()
            if args.features_scaling:
                test_data.normalize_features(features_scaler)
            if train_args.atom_descriptor_scaling and args.atom_descriptors is not None:
                test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
            if train_args.bond_feature_scaling and args.bond_features_size > 0:
                test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

        # Make predictions
        model_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=scaler
        )
        if args.dataset_type == 'spectra':
            model_preds = normalize_spectra(
                spectra=model_preds,
                phase_features=test_data.phase_features(),
                phase_mask=args.spectra_phase_mask,
                excluded_sub_value=float('nan')
            )
        sum_preds += np.array(model_preds)
        if args.ensemble_variance or args.individual_ensemble_predictions:
            if args.dataset_type == 'multiclass':
                all_preds[:, :, :, index] = model_preds
            else:
                all_preds[:, :, index] = model_preds

    # Ensemble predictions
    avg_preds = sum_preds / len(args.checkpoint_paths)

    if args.ensemble_variance:
        if args.dataset_type == 'spectra':
            all_epi_uncs = roundrobin_sid(all_preds)
        else:
            all_epi_uncs = np.var(all_preds, axis=2)
            all_epi_uncs = all_epi_uncs.tolist()

    # Save predictions
    # print(f'Saving predictions to {args.preds_path}')
    assert len(test_data) == len(avg_preds)
    if args.ensemble_variance:
        assert len(test_data) == len(all_epi_uncs)
    makedirs(args.preds_path, isfile=True)

    # Set multiclass column names, update num_tasks definition for multiclass
    if args.dataset_type == 'multiclass':
        task_names = [f'{name}_class_{i}' for name in task_names for i in range(args.multiclass_num_classes)]
        num_tasks = num_tasks * args.multiclass_num_classes

    # Copy predictions over to full_data
    for full_index, datapoint in enumerate(full_data):
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = avg_preds[valid_index] if valid_index is not None else ['Invalid SMILES'] * num_tasks
        if args.ensemble_variance:
            if args.dataset_type == 'spectra':
                epi_uncs = all_epi_uncs[valid_index] if valid_index is not None else ['Invalid SMILES']
            else:
                epi_uncs = all_epi_uncs[valid_index] if valid_index is not None else ['Invalid SMILES'] * num_tasks
        if args.individual_ensemble_predictions:
            ind_preds = all_preds[valid_index] if valid_index is not None else [['Invalid SMILES'] * len(
                args.checkpoint_paths)] * num_tasks

        # Reshape multiclass to merge task and class dimension, with updated num_tasks
        if args.dataset_type == 'multiclass':
            preds = preds.reshape((num_tasks))
            if args.ensemble_variance or args.individual_ensemble_predictions:
                ind_preds = ind_preds.reshape((num_tasks, len(args.checkpoint_paths)))

        # If extra columns have been dropped, add back in SMILES columns
        if args.drop_extra_columns:
            datapoint.row = OrderedDict()

            smiles_columns = args.smiles_columns

            for column, smiles in zip(smiles_columns, datapoint.smiles):
                datapoint.row[column] = smiles

        # Add predictions columns
        for pred_name, pred in zip(task_names, preds):
            datapoint.row[pred_name] = pred
        if args.individual_ensemble_predictions:
            for pred_name, model_preds in zip(task_names, ind_preds):
                for idx, pred in enumerate(model_preds):
                    datapoint.row[pred_name + f'_model_{idx}'] = pred
        if args.ensemble_variance:
            if args.dataset_type == 'spectra':
                datapoint.row['epi_unc'] = epi_uncs
            else:
                for pred_name, epi_unc in zip(task_names, epi_uncs):
                    datapoint.row[pred_name + '_epi_unc'] = epi_unc

    # Save
    # with open(args.preds_path, 'w') as f:
    #     writer = csv.DictWriter(f, fieldnames=full_data[0].row.keys())
    #     writer.writeheader()
    #
    #     for datapoint in full_data:
    #         writer.writerow(datapoint.row)

    avg_preds = avg_preds.tolist()

    return avg_preds

def calculate_chemprop_active(args: PredictArgs, gen_molecule : List[List[str]],
                     model_objects: Tuple[PredictArgs, TrainArgs, List[MoleculeModel], List[StandardScaler], int, List[str]] = None):
    if model_objects:
        args, train_args, models, scalers, num_tasks, task_names = model_objects
    else:
        args, train_args, models, scalers, num_tasks, task_names = load_model(args, generator=True)

    set_features(args, train_args)

    full_data, test_data, test_data_loader, full_to_valid_indices = load_data(args, gen_molecule)

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
    #     return [None] * len(full_data)
        return 0.0

    avg_preds = predict_and_save(args, train_args, test_data, task_names, num_tasks, test_data_loader, full_data,
                                 full_to_valid_indices, models, scalers)

    pred_float = avg_preds[0][0]
    return pred_float


def check_for_nan(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

def replace_nan(tensor, value=0.0):
    if torch.isnan(tensor).any():
        # print("NaN detected in tensor, replacing with value:", value)
        tensor = torch.nan_to_num(tensor, nan=value)
    return tensor

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reinitialize_model(model):
    # 重新初始化 layer1
    if hasattr(model, 'layer1') and hasattr(model.layer1, 'weight'):
        nn.init.xavier_uniform_(model.layer1.weight)
        if model.layer1.bias is not None:
            model.layer1.bias.data.fill_(0.0)

    # 重新初始化 layer2
    if hasattr(model, 'layer2') and hasattr(model.layer2, 'weight'):
        nn.init.xavier_uniform_(model.layer2.weight)
        if model.layer2.bias is not None:
            model.layer2.bias.data.fill_(0.0)

    # 重新初始化 layer3
    if hasattr(model, 'layer3') and hasattr(model.layer3, 'weight'):
        nn.init.xavier_uniform_(model.layer3.weight)
        if model.layer3.bias is not None:
            model.layer3.bias.data.fill_(0.0)

def remove_nan(X, Y):
    nan_mask = torch.isnan(X).any(dim=1) | torch.isnan(Y).any(dim=1)
    X_clean = X[~nan_mask]
    Y_clean = Y[~nan_mask]
    return X_clean, Y_clean

def aac_comp(sequences):
    std = list("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acids
    features = []
    for seq in sequences:
        composition = [seq.count(aa) / len(seq) * 100 for aa in std]
        features.append(composition)
    return np.array(features)




def dpc_comp(sequences, q=1):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    features = []
    for seq in sequences:
        cc = []
        for j in std:
            for k in std:
                count = sum(1 for m in range(len(seq) - q) if seq[m:m + q + 1:q].upper() == j + k)
                cc.append(count / (len(seq) - q) * 100 if (len(seq) - q) > 0 else 0)
        features.append(cc)
    return np.array(features)



def predict_scores(aac_features, dpc_features, model_path='/root/morbo/toxinpred3-main/model/toxinpred3.0_model.pkl'):
    clf = joblib.load(model_path)
    features = np.hstack((aac_features, dpc_features))
    predictions = clf.predict_proba(features)[:, 1]  # assuming the second column for positive class
    return 1-predictions

def calculate_peptide_toxic(sequences, model_path='/root/morbo/toxinpred3-main/model/toxinpred3.0_model.pkl'):
    aac_features = aac_comp([sequences])
    dpc_features = dpc_comp([sequences])
    scores = predict_scores(aac_features, dpc_features, model_path)
    single_score = float(scores[0])
    return -single_score