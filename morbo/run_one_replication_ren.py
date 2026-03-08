#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Run one replication.
"""
from typing import Callable, Dict, List, Optional, Union
import time
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.test_functions import Ackley
from botorch.test_functions.multi_objective import (
    BraninCurrin,
    C2DTLZ2,
    DH2,
    DH3,
    DH4,
    DTLZ1,
    DTLZ2,
    MW7,
    VehicleSafety,
    WeldedBeam,
)

from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from .get_init import init_x,init_y
from morbo.gen import (
    TS_select_batch_MORBO,
    Greed_select_psl,
)
from morbo.state import TRBOState
from morbo.trust_region import TurboHParams
from torch import Tensor

from morbo.problems.rover import get_rover_fn
from morbo.problems.mol_ren import get_mol_fn, check_for_nan, replace_nan, set_seed,reinitialize_model,get_mol_fn_peptides,get_mol_fn_EP
from morbo.benchmark_function import (
    BenchmarkFunction,
)
from .lolbo.molecule_objective import MoleculeObjective
from morbo.pslmodel_ren import ParetoSetModel256, ParetoSetModel
import numpy as np
import warnings
from .sample_mol import read_csv_column
from .chemVAE import main
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from transvae.trans_models import TransVAE
from botorch.utils.transforms import normalize,unnormalize

warnings.filterwarnings("ignore")

supported_labels = ["morbo"]

BASE_SEED = 12345

def run_one_replication_ren(
    seed: int,
    label: str,
    max_evals: int,
    evalfn: str,
    batch_size: int,
    dim: int,
    n_initial_points: int,
    n_trust_regions: int = TurboHParams.n_trust_regions,
    max_tr_size: int = TurboHParams.max_tr_size,
    min_tr_size: int = TurboHParams.min_tr_size,
    max_reference_point: Optional[List[float]] = None,
    failure_streak: Optional[int] = None,  # This is better to set automatically
    success_streak: int = TurboHParams.success_streak,
    raw_samples: int = TurboHParams.raw_samples,
    n_restart_points: int = TurboHParams.n_restart_points,
    length_init: float = TurboHParams.length_init,
    length_min: float = TurboHParams.length_min,
    length_max: float = TurboHParams.length_max,
    trim_trace: bool = TurboHParams.trim_trace,
    hypervolume: bool = TurboHParams.hypervolume,
    max_cholesky_size: int = TurboHParams.max_cholesky_size,
    use_ard: bool = TurboHParams.use_ard,
    verbose: bool = TurboHParams.verbose,
    qmc: bool = TurboHParams.qmc,
    track_history: bool = TurboHParams.track_history,
    sample_subset_d: bool = TurboHParams.sample_subset_d,
    fixed_scalarization: bool = TurboHParams.fixed_scalarization,
    winsor_pct: float = TurboHParams.winsor_pct,
    trunc_normal_perturb: bool = TurboHParams.trunc_normal_perturb,
    switch_strategy_freq: Optional[int] = TurboHParams.switch_strategy_freq,
    tabu_tenure: int = TurboHParams.tabu_tenure,
    decay_restart_length_alpha: float = TurboHParams.decay_restart_length_alpha,
    use_noisy_trbo: bool = TurboHParams.use_noisy_trbo,
    observation_noise_std: Optional[List[float]] = None,
    observation_noise_bias: Optional[List[float]] = None,
    use_simple_rff: bool = TurboHParams.use_simple_rff,
    use_approximate_hv_computations: bool = TurboHParams.use_approximate_hv_computations,
    approximate_hv_alpha: Optional[float] = TurboHParams.approximate_hv_alpha,
    recompute_all_hvs: bool = True,
    restart_hv_scalarizations: bool = True,
    dtype: torch.device = torch.double,
    device: Optional[torch.device] = None,
    save_callback: Optional[Callable[[Tensor], None]] = None,
    save_during_opt: bool = True,
    n_steps: int = 1000,
    n_pref_update: int = 100,
    num_candidate: int = 1000,
) -> None:
    r"""Run the BO loop for given number of iterations. Supports restarting of
    prematurely killed experiments.

    Args:
        seed: The random seed.
        label: The algorith ("morbo")
        max_evals: evaluation budget
        evalfn: The test problem name
        batch_size: The size of each batch in BO
        dim: The input dimension (this is a parameter for some problems)
        n_initial_points: The number of initial sobol points

    The remaining parameters and default values are defined in trust_region.py.
    """
    assert label in supported_labels, "Label not supported!"
    start_time = time.time()

    seed = BASE_SEED + seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    seed = set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tkwargs = {"dtype": dtype, "device": device}
    bounds = torch.empty((2, dim), dtype=dtype, device=device)
    constraints = None
    objective = None

    if evalfn == "ackley":
        f = Ackley(dim=dim, negate=False)
        bounds[0] = -5.0
        bounds[1] = 10.0
        num_outputs = 1
        num_objectives = 1
    elif max_reference_point is None:
        raise ValueError(f"max_reference_point is required for {evalfn}")
    else:
        num_objectives = len(max_reference_point)

    if evalfn == "C2DTLZ2":
        problem = C2DTLZ2(
            num_objectives=num_objectives,
            dim=dim,
            negate=False,
        )
        bounds = problem.bounds.to(**tkwargs)
        num_outputs = problem.num_objectives + problem.num_constraints
        # Note: all outcomes are multiplied by -1 in `BenchmarkFunction` by default.
        constraints = (
            torch.tensor([[0.0, 0.0, 1.0]], **tkwargs),
            torch.tensor([[0.0]], **tkwargs),
        )

        def f(X):
            return torch.cat([problem(X), problem.evaluate_slack(X)], dim=-1)

        objective = IdentityMCMultiOutputObjective(
            outcomes=[0, 1], num_outcomes=num_outputs
        )
    elif evalfn == "rover":
        num_objectives, num_outputs = 2, 2
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}.")
        f, bounds = get_rover_fn(
            dim, device=device, dtype=dtype, force_goal=False, force_start=True
        )
    elif evalfn =="mol_selfies":
        num_objectives, num_outputs = 2, 2
        # pkl_path = '/root/morbo/morbo/chemVAE/model_2024-01-08_parameters_epoch100.pkl'
        # ## load parameters
        # pkl_file = open(pkl_path, 'rb')
        # params = pickle.load(pkl_file)
        # pkl_file.close()

        # ## load model state
        # model = main.VAE(params).to(device)
        # #model_state_path = '/home/leo/fightinglee/Antibiotic-project/chemVAE-main/chemVAE/model_2023-12-11_state_epoch300.pth'
        # model_state_path = '/root/morbo/morbo/chemVAE/model_2024-01-08_state_epoch100.pth'
        # model.load_state_dict(torch.load(model_state_path))

        # df = pd.read_csv('/root/morbo/morbo/TRAIN_VAE_dataset.csv')
        # smiles = df['SMILES'].values
        # selfies = main.smiles2selfies(smiles)
        # onehot_selfies, idx_to_symbol = main.onehotSELFIES(selfies)

        # ckpt_fn = '/root/autodl-tmp/checkpoints/050_2M_4x256.ckpt'
        # ckpt_fn = '/root/autodl-tmp/checkpoints/010_7M_model.ckpt'
        # ckpt_fn = '/root/autodl-tmp/checkpoints/020_10M_4X256_model.ckpt'
        # ckpt_fn = '/root/autodl-tmp/checkpoints/035_5M_4X256_model.ckpt'
        # ckpt_fn = '/root/autodl-tmp/checkpoints/050_5M_4X256_model.ckpt'
        # ckpt_fn = '/root/morbo/TransVAE-master/ckpt/040_antibiotic_all_model.ckpt'
        ckpt_fn = '/root/morbo/TransVAE-master/ckpt/060_antibiotic_all_model.ckpt'
        vae = TransVAE(load_fn=ckpt_fn)

        model = vae

        #encoder-decoder
        # model = MoleculeObjective(task_id = 'logp')
        f, bounds = get_mol_fn(
            dim, model,device=device, dtype=dtype,
        )

    elif evalfn =="EP":
        num_objectives, num_outputs = 2, 2
        # pkl_path = '/root/morbo/morbo/chemVAE/model_2024-01-08_parameters_epoch100.pkl'
        # ## load parameters
        # pkl_file = open(pkl_path, 'rb')
        # params = pickle.load(pkl_file)
        # pkl_file.close()

        # ## load model state
        # model = main.VAE(params).to(device)
        # #model_state_path = '/home/leo/fightinglee/Antibiotic-project/chemVAE-main/chemVAE/model_2023-12-11_state_epoch300.pth'
        # model_state_path = '/root/morbo/morbo/chemVAE/model_2024-01-08_state_epoch100.pth'
        # model.load_state_dict(torch.load(model_state_path))

        # df = pd.read_csv('/root/morbo/morbo/TRAIN_VAE_dataset.csv')
        # smiles = df['SMILES'].values
        # selfies = main.smiles2selfies(smiles)
        # onehot_selfies, idx_to_symbol = main.onehotSELFIES(selfies)

        # ckpt_fn = '/root/autodl-tmp/checkpoints/050_2M_4x256.ckpt'
        # ckpt_fn = '/root/autodl-tmp/checkpoints/010_7M_model.ckpt'
        # ckpt_fn = '/root/autodl-tmp/checkpoints/020_10M_4X256_model.ckpt'
        # ckpt_fn = '/root/autodl-tmp/checkpoints/035_5M_4X256_model.ckpt'
        # ckpt_fn = '/root/autodl-tmp/checkpoints/050_5M_4X256_model.ckpt'
        ckpt_fn = '/root/morbo/TransVAE-master/ckpt/040_antibiotic_all_model.ckpt'
        vae_EP = TransVAE(load_fn=ckpt_fn)

        model = vae_EP

        #encoder-decoder
        # model = MoleculeObjective(task_id = 'logp')
        f, bounds = get_mol_fn_EP(
            dim, model,device=device, dtype=dtype,
        )
    elif evalfn == "peptides":
        num_objectives, num_outputs = 2, 2

        pkl_path = '/root/morbo/morbo/chemVAE/model_2024-06-29_parameters_peptides.pkl'
        ## load parameters
        pkl_file = open(pkl_path, 'rb')
        params = pickle.load(pkl_file)
        pkl_file.close()

        ## load model state
        model_peptides = main.VAE(params).to(device)
        #model_state_path = '/home/leo/fightinglee/Antibiotic-project/chemVAE-main/chemVAE/model_2023-12-11_state_epoch300.pth'
        model_state_path = '/root/morbo/morbo/chemVAE/model_2024-06-29_state_epoch_peptides.pth'
        model_peptides.load_state_dict(torch.load(model_state_path))

        df = pd.read_csv('/root/morbo/TransVAE-master/data/peptides/E.coli-AMP.csv')
        peptides = df['sequences'].values
        onehot_peptides, idx_to_symbol = main.onehot_amino_acids(peptides)

        f, bounds = get_mol_fn_peptides(
            dim, model_peptides, idx_to_symbol, device=device, dtype=dtype
        )


    elif evalfn == "WeldedBeam":
        problem = WeldedBeam(negate=False)
        bounds = problem.bounds.to(**tkwargs)

        def f(X):
            return torch.cat([problem(X), problem.evaluate_slack(X)], dim=-1)

        num_objectives = 2
        num_constraints = 4
        num_outputs = num_objectives + num_constraints

        Z_ = torch.zeros(num_constraints, num_objectives, **tkwargs)
        A = torch.cat((Z_, torch.eye(num_constraints, **tkwargs)), dim=1)
        constraints = (A, torch.zeros(num_constraints, 1, **tkwargs))

        objective = IdentityMCMultiOutputObjective(
            outcomes=[0, 1], num_outcomes=num_outputs
        )
    elif evalfn == "MW7":
        problem = MW7(negate=False, dim=dim)
        bounds = problem.bounds.to(**tkwargs)

        def f(X):
            return torch.cat([problem(X), problem.evaluate_slack(X)], dim=-1)

        num_objectives = 2
        num_constraints = 2
        num_outputs = num_objectives + num_constraints

        Z_ = torch.zeros(num_constraints, num_objectives, **tkwargs)
        A = torch.cat((Z_, torch.eye(num_constraints, **tkwargs)), dim=1)
        constraints = (A, torch.zeros(num_constraints, 1, **tkwargs))

        objective = IdentityMCMultiOutputObjective(
            outcomes=[0, 1], num_outcomes=num_outputs
        )
    elif evalfn != "ackley":
        # Handle the non-constrained botorch test functions here.
        constructor_map = {
            "DH2": DH2,
            "DH3": DH3,
            "DH4": DH4,
            "DTLZ1": DTLZ1,
            "DTLZ2": DTLZ2,
            "BraninCurrin": BraninCurrin,
            "VehicleSafety": VehicleSafety,
        }
        constructor_args = {"negate": False}
        if evalfn not in ("BraninCurrin", "VehicleSafety"):
            constructor_args["dim"] = dim
        if "DTLZ" in evalfn:
            constructor_args["num_objectives"] = num_objectives
        if evalfn not in constructor_map:
            raise ValueError("Unknown `evalfn` specified!")
        f = constructor_map[evalfn](**constructor_args)
        bounds = f.bounds.to(**tkwargs)
        num_outputs = f.num_objectives
    #f是评估函数
    f = BenchmarkFunction(
        base_f=f,
        num_outputs=num_outputs,
        ref_point=torch.tensor(max_reference_point, **tkwargs),
        dim=dim,
        tkwargs=tkwargs,
        negate=True,
        observation_noise_std=observation_noise_std,
        observation_noise_bias=observation_noise_bias,
    )

    # Automatically set the failure streak if it isn't specified
    failure_streak = max(dim // 3, 10) if failure_streak is None else failure_streak
    #tr_hparams是类定义了TuRBO（信任区域贝叶斯优化）算法的超参数和设置
    tr_hparams = TurboHParams(
        length_init=length_init,
        length_min=length_min,
        length_max=length_max,
        batch_size=batch_size,
        success_streak=success_streak,
        failure_streak=failure_streak,
        max_tr_size=max_tr_size,
        min_tr_size=min_tr_size,
        trim_trace=trim_trace,
        n_trust_regions=n_trust_regions,
        verbose=verbose,
        qmc=qmc,
        use_ard=use_ard,
        sample_subset_d=sample_subset_d,
        track_history=track_history,
        fixed_scalarization=fixed_scalarization,
        n_initial_points=n_initial_points,
        n_restart_points=n_restart_points,
        raw_samples=raw_samples,
        max_reference_point=max_reference_point,
        hypervolume=hypervolume,
        winsor_pct=winsor_pct,
        trunc_normal_perturb=trunc_normal_perturb,
        decay_restart_length_alpha=decay_restart_length_alpha,
        switch_strategy_freq=switch_strategy_freq,
        tabu_tenure=tabu_tenure,
        use_noisy_trbo=use_noisy_trbo,
        use_simple_rff=use_simple_rff,
        use_approximate_hv_computations=use_approximate_hv_computations,
        approximate_hv_alpha=approximate_hv_alpha,
        restart_hv_scalarizations=restart_hv_scalarizations,
        n_pref_update = n_pref_update,
        n_steps = n_steps,
    )
#信任区域贝叶斯优化（TuRBO）算法
    trbo_state = TRBOState(
        dim=dim,
        max_evals=max_evals,
        num_outputs=num_outputs,
        num_objectives=num_objectives,
        bounds=bounds,
        tr_hparams=tr_hparams,
        constraints=constraints,
        objective=objective,
    )

    # For saving outputs
    n_evals = []
    true_hv = []
    pareto_X = []
    pareto_Y = []
    n_points_in_tr = [[] for _ in range(n_trust_regions)]
    n_points_in_tr_collected_by_other = [[] for _ in range(n_trust_regions)]
    n_points_in_tr_collected_by_sobol = [[] for _ in range(n_trust_regions)]
    tr_sizes = [[] for _ in range(n_trust_regions)]
    tr_centers = [[] for _ in range(n_trust_regions)]
    tr_restarts = [[] for _ in range(n_trust_regions)]
    fit_times = []
    gen_times = []
    true_ref_point = torch.tensor(max_reference_point, dtype=dtype, device=device)

    # Create initial points
    n_points = min(n_initial_points, max_evals - trbo_state.n_evals)
    #smiles_init = read_csv_column('/root/morbo/morbo/train-negative_clean_token.csv','SMILES',6187)

    # #04-17
    df = pd.read_csv('/root/morbo/TransVAE-master/data/antibiotic/FDA_Gram-Positive_Bacteria_Antibiotics.csv')
    smiles_init = df['SMILES'].values

    #peptides
    # df = pd.read_csv('/root/morbo/TransVAE-master/AFP-svm/data/filtered_uniprot.csv')
    # smiles_init = df['Sequence'].values

    X_init = []
    # for smiles in smiles_init:
    #     # X, _ = vae.reconstruct_encoder([smiles])
    #     # X,_ = model.vae_forward([smiles])
    #     onehot_selfies, idx_to_symbol = main.onehot_amino_acids([smiles])
    #     data_loader = DataLoader(onehot_selfies, batch_size=1, shuffle=False)
    #     for batch in data_loader:
    #         batch = batch.to(device)
    #         with torch.no_grad():
    #             X, _, _ = model_peptides.encoder(batch)
    #     X_init.append(X)

    X_history_smiles = []
    for smiles in smiles_init:
        X, _ = vae.reconstruct_encoder([smiles])
        X_init.append(X)
    X_init = torch.cat(X_init)
    X_init = X_init.detach()
    X_init = X_init.squeeze(1).to(torch.float64)
    Y_init,dec_smiles = f(X_init)
    X_history_smiles.extend(dec_smiles)

    trbo_state.update(
        X=X_init,
        Y=Y_init,
        new_ind=torch.full(
            (X_init.shape[0],), 0, dtype=torch.long, device=X_init.device
        ),
    )
    trbo_state.log_restart_points(X=X_init, Y=Y_init)

    # Initializing the trust regions. This also initializes the models.
    for i in range(n_trust_regions):
        trbo_state.initialize_standard(
            tr_idx=i,
            restart=False,
            switch_strategy=False,
            X_init=X_init,
            Y_init=Y_init,
        )

    # Update TRs data across trust regions, if necessary
    trbo_state.update_data_across_trs()

    # Set the initial TR indices to -2
    trbo_state.TR_index_history.fill_(-2)

    # Getting next suggestions
    all_tr_indices = [-1] * n_points
    restart_counter = 0
    z = torch.tensor([1,1]).to(device)
    X, Y = X_init,Y_init

    global_solution = []
    while trbo_state.n_evals < max_evals:
        print("Creating PSL model...")
        pslmodel = ParetoSetModel256(num_objectives, hidden_dim=256, output_dim = dim, num_heads=4,device='cuda:0')
        pslmodel.to(device)

        # print("Initializing optimizer...")
        optimizer = torch.optim.Adam(pslmodel.parameters(), lr=1e-4)

        z = torch.max(torch.cat((z.reshape(1, num_objectives), replace_nan(Y, 0.0) + 0.1)), axis=0).values.data

        trbo_state.set_utopia(z)

        # print("Training the model...")
        trbo_state.set_pslmodel(optimizer=optimizer, pslmodel=pslmodel)
        start_gen = time.time()


        # print("Updating model state...")

        trbo_state.update_pslmodel()


        alpha = np.ones(num_objectives)
        pref = np.random.dirichlet(alpha,num_candidate)
        pref  = torch.tensor(pref).to(device).float() + 0.0001
        pref = pref.to(device)
        X_cand_pslmodel = trbo_state.pslmodel(pref)
        X_cand_pslmodel = torch.unique(X_cand_pslmodel, dim=0)
        X_cand_pslmodel = unnormalize(X_cand_pslmodel, bounds=bounds)

        print("generating PSL candidates")
        #筛选全局模型候选物
        psl_selected  = Greed_select_psl(trbo_state = trbo_state, psl_candidate=X_cand_pslmodel)
        gen_times.append(time.time() - start_gen)
        if trbo_state.tr_hparams.verbose:
            print(f"Time spent on generating candidates: {gen_times[-1]:.1f} seconds")
        start_gen = time.time()

        psl_cand = psl_selected.X_cand
        psl_cand = psl_cand.detach()
        psl_indices =psl_selected.tr_indices

        assert len(torch.unique(psl_cand,dim = 0)) == len(psl_cand)
        #筛选局部模型候选物
        selection_output = TS_select_batch_MORBO(trbo_state=trbo_state)
        gen_times.append(time.time() - start_gen)
        if trbo_state.tr_hparams.verbose:
            print(f"Time spent on generating candidates: {gen_times[-1]:.1f} seconds")
        
        #create Candidate by pslmodel
        X_cand = selection_output.X_cand
        #assert len(torch.unique(X_cand,dim = 0)) == len(X_cand)
        X_cand = torch.cat([psl_cand,X_cand], dim = 0)
        # X_cand = torch.unique(X_cand,dim=0)
        tr_indices = selection_output.tr_indices
        tr_indices = torch.cat((psl_indices,tr_indices),dim = -1)
        # all_tr_indices.extend(tr_indices.tolist())
        # trbo_state.tabu_set.log_iteration()

        Y_cand, dec_smiles = f(X_cand)
        mask_0 = torch.all(Y_cand == 0, dim=1)
        X_cand = X_cand[~mask_0]
        Y_cand = Y_cand[~mask_0]
        # print('Shape of X_cand:', X_cand.shape)
        tr_indices = tr_indices[~mask_0]  # 过滤tr_indices以保持一致性
        # 将过滤后的 tr_indices 添加到 all_tr_indices 列表
        filtered_smiles = [smile for i, smile in enumerate(dec_smiles) if not mask_0[i]]

        mask_smiles = [smile in X_history_smiles for smile in filtered_smiles]
        X_cand = X_cand[~torch.tensor(mask_smiles)]
        Y_cand = Y_cand[~torch.tensor(mask_smiles)]
        tr_indices = tr_indices[~torch.tensor(mask_smiles)]
        X_history_smiles.extend([smile for i, smile in enumerate(filtered_smiles) if not mask_smiles[i]])


        all_tr_indices.extend(tr_indices.tolist())
        # 记录迭代情况
        trbo_state.tabu_set.log_iteration()

        # Log TR info
        for i, tr in enumerate(trbo_state.trust_regions):
            inds = torch.cat(
                [torch.where((x == trbo_state.X_history).all(dim=-1))[0] for x in tr.X])
            inds_unique = torch.unique(inds)
            tr_inds = trbo_state.TR_index_history[inds_unique]
            # print(f"tr_inds:{tr_inds}")
            # print(f"tr_X:{tr.X}")
            assert len(tr_inds) == len(tr.X)
            n_points_in_tr[i].append(len(tr_inds))
            n_points_in_tr_collected_by_sobol[i].append(sum(tr_inds == -2).cpu().item())
            n_points_in_tr_collected_by_other[i].append(
                sum((tr_inds != i) & (tr_inds != -2)).cpu().item()
            )
            tr_sizes[i].append(tr.length.item())
            tr_centers[i].append(tr.X_center.cpu().squeeze().tolist())

        # Append data to the global history and fit new models
        start_fit = time.time()

        trbo_state.update(X=X_cand, Y=Y_cand, new_ind=tr_indices)
        # trbo_state.X_history = replace_nan(trbo_state.X_history, 0.0)
        should_restart_trs = trbo_state.update_trust_regions_and_log(
            X_cand=X_cand,
            Y_cand=Y_cand,
            tr_indices=tr_indices,
            batch_size=batch_size,
            #batch_size=len(tr_indices),
            verbose=verbose,
        )
        fit_times.append(time.time() - start_fit)
        if trbo_state.tr_hparams.verbose:
            print(f"Time spent on model fitting: {fit_times[-1]:.1f} seconds")

        switch_strategy = trbo_state.check_switch_strategy()
        if switch_strategy:
            should_restart_trs = [True for _ in should_restart_trs]
        if any(should_restart_trs):
            for i in range(trbo_state.tr_hparams.n_trust_regions):
                if should_restart_trs[i]:
                    restart_counter += 1
                    n_points = min(n_restart_points, max_evals - trbo_state.n_evals)
                    if n_points <= 0:
                        break  # out of budget
                    if trbo_state.tr_hparams.verbose:
                        print(f"{trbo_state.n_evals}) Restarting trust region {i}")
                    trbo_state.TR_index_history[trbo_state.TR_index_history == i] = -1
                    init_kwargs = {}
                    if trbo_state.tr_hparams.restart_hv_scalarizations:
                        # generate new point
                        X_center = trbo_state.gen_new_restart_design()
                        Y_center, _= f(X_center)
                        init_kwargs["X_init"] = X_center
                        init_kwargs["Y_init"] = Y_center
                        init_kwargs["X_center"] = X_center
                        trbo_state.update(
                            X=X_center,
                            Y=Y_center,
                            new_ind=torch.tensor(
                                [i], dtype=torch.long, device=X_center.device
                            ),
                        )
                        trbo_state.log_restart_points(X=X_center, Y=Y_center)

                    trbo_state.initialize_standard(
                        tr_idx=i,
                        restart=True,
                        switch_strategy=switch_strategy,
                        **init_kwargs,
                    )
                    if trbo_state.tr_hparams.restart_hv_scalarizations:
                        # we initialized the TR with one data point.
                        # this passes historical information to that new TR
                        trbo_state.update_data_across_trs()
                    tr_restarts[i].append(
                        trbo_state.n_evals.item()
                    )  # Where it restarted

        if trbo_state.tr_hparams.verbose:
            print(f"Total refill points: {trbo_state.total_refill_points}")

        # Save state at this evaluation and move to cpu
        n_evals.append(trbo_state.n_evals.item())
        if trbo_state.hv is not None:
            # The objective is None if there are no constraints
            obj = objective if objective else lambda x: x
            partitioning = DominatedPartitioning(
                ref_point=true_ref_point, Y=obj(trbo_state.pareto_Y)
            )
            hv = partitioning.compute_hypervolume().item()
            if trbo_state.tr_hparams.verbose:
                print(f"{trbo_state.n_evals}) Current hypervolume: {hv:.3f}")

            pareto_X.append(trbo_state.pareto_X.tolist())
            pareto_Y.append(trbo_state.pareto_Y.tolist())
            true_hv.append(hv)

            if observation_noise_std is not None:
                f.record_current_pf_and_hv(obj=obj, constraints=trbo_state.constraints)
        else:
            if trbo_state.tr_hparams.verbose:
                print(f"{trbo_state.n_evals}) Current hypervolume is zero!")
            pareto_X.append([])
            pareto_Y.append([])
            true_hv.append(0.0)

        trbo_state.update_data_across_trs()


        output = {
            "n_evals": n_evals,
            "X_history": trbo_state.X_history.cpu(),
            "metric_history": trbo_state.Y_history.cpu(),
            "true_pareto_X": pareto_X,
            "true_pareto_Y": pareto_Y,
            "true_hv": true_hv,
            "n_points_in_tr": n_points_in_tr,
            "n_points_in_tr_collected_by_other": n_points_in_tr_collected_by_other,
            "n_points_in_tr_collected_by_sobol": n_points_in_tr_collected_by_sobol,
            "tr_sizes": tr_sizes,
            "tr_centers": tr_centers,
            "tr_restarts": tr_restarts,
            "fit_times": fit_times,
            "gen_times": gen_times,
            "tr_indices": all_tr_indices,
            "X_history_smiles":X_history_smiles,
        }
        # Save the output.
        if save_during_opt is not None:
            save_callback(output)
        
    end_time = time.time()
    if trbo_state.tr_hparams.verbose:
        print(f"Total time: {end_time - start_time:.1f} seconds")

    if trbo_state.hv is not None and recompute_all_hvs:
        # Go back and compute all hypervolumes so we don't have to do that later...
        f.record_all_hvs(obj=obj, constraints=trbo_state.constraints)

    output = {
        "n_evals": n_evals,
        "X_history": trbo_state.X_history.cpu(),
        "metric_history": trbo_state.Y_history.cpu(),
        "true_pareto_X": pareto_X,
        "true_pareto_Y": pareto_Y,
        "true_hv": true_hv,
        "n_points_in_tr": n_points_in_tr,
        "n_points_in_tr_collected_by_other": n_points_in_tr_collected_by_other,
        "n_points_in_tr_collected_by_sobol": n_points_in_tr_collected_by_sobol,
        "tr_sizes": tr_sizes,
        "tr_centers": tr_centers,
        "tr_restarts": tr_restarts,
        "fit_times": fit_times,
        "gen_times": gen_times,
        "tr_indices": all_tr_indices,
        "X_history_smiles": X_history_smiles,
    }
    if trbo_state.hv is not None and recompute_all_hvs:
        additional_outputs = f.get_outputs()
        output = {**output, **additional_outputs}
    
    # alpha = np.ones(num_objectives)
    # pref = np.random.dirichlet(alpha,2500)
    # pref  = torch.tensor(pref).to(device).float() + 0.0001
    # pref = pref.to(device)
    # Generated_x = trbo_state.pslmodel(pref).to(torch.float)
    # _,Generated_smile = model.vae_decode(Generated_x)
    # pareto_mol_x = torch.tensor(pareto_X[-1])
    # pareto_mol_x = pareto_mol_x.to(torch.float)
    # _,Generated_smiles_local = model.vae_decode(pareto_mol_x)
    # print(f"len:{len(Generated_smiles_local)}")
    # Generated_smile = Generated_smile + Generated_smiles_local
    # fn = '/root/morbo/result/Generated_smile_'+str(seed-BASE_SEED)+'.txt'
    # file = open(fn, 'w')
    # for item in Generated_smile:
    #     file.write(item+'\n')
    # file.close()
    # Generated_y = f(Generated_x)
    # Generated_y = Generated_y.to('cpu')
    # fn = '/root/morbo/result/Generated_y_'+str(seed-BASE_SEED)+'.pt'
    # torch.save(Generated_y,fn)
    # global_smiles = []
    # for i in range(len(global_solution)):
    #     _,global_smile = model.vae_decode(global_solution[i])
    #     for j in range(len(global_smile)):
    #         global_smiles.append(global_smile[j])
    # fn = '/root/morbo/result/global_smile_'+str(seed-BASE_SEED)+'.txt'
    # file = open(fn, 'w')
    # for item in global_smiles:
    #     file.write(item+'\n')
    # file.close()
    fn = '/root/morbo/result/model'+str(seed)+'.pt'
    torch.save(trbo_state.pslmodel,fn)
    #trained_model = trbo_state.pslmodel
    
    # Save the final output
    save_callback(output)
