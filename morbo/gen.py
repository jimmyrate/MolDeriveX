#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import time
from typing import Callable, NamedTuple, Tuple
import random
import gpytorch
import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.gp_sampling import get_gp_samples
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import normalize, unnormalize
from morbo.state import TRBOState
from morbo.utils import (
    decay_function,
    get_indices_in_hypercube,
    sample_tr_discrete_points,
    get_indices_closest_hypercube,
    sample_tr_discrete_points_subset_d,
)
from torch import Tensor
from torch.quasirandom import SobolEngine
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.performance_indicator.hv import Hypervolume as HV
import numpy as np
from scipy.stats import norm

CUDA_LAUNCH_BLOCKING=1

class CandidateSelectionOutput(NamedTuple):
    X_cand: Tensor
    tr_indices: Tensor


def get_partitioning(
    trbo_state: TRBOState, ref_point: Tensor, Y: Tensor
) -> BoxDecomposition:
    """Helper method for constructing a box decomposition"""
    if trbo_state.tr_hparams.use_approximate_hv_computations:
        alpha = (
            trbo_state.tr_hparams.approximate_hv_alpha
            if trbo_state.tr_hparams.approximate_hv_alpha is not None
            else get_default_partitioning_alpha(trbo_state.num_objectives)
        )
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y, alpha=alpha)
    else:
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=Y)
    return partitioning


def _make_unstandardizer(Y_mean: Tensor, Y_std: Tensor) -> Callable[[Tensor], Tensor]:
    def unstandardizer(Y: Tensor) -> Tensor:
        return Y * Y_std + Y_mean

    return unstandardizer


def preds_and_feas(
    trbo_state: TRBOState, tr_idx: int, X: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute model predictions and constraint violations."""
    tkwargs = {"device": trbo_state.bounds.device, "dtype": trbo_state.bounds.dtype}
    tr = trbo_state.trust_regions[tr_idx]
    objective = tr.objective
    model = trbo_state.models[tr_idx]
    preds, dists = model.get_predictions_and_distances(X)
    # apply objective
    f_obj = objective(preds).clone()

    if trbo_state.constraints is not None:
        constraint_value = torch.stack(
            [c(preds) for c in trbo_state.constraints], dim=-1
        )
        feas = (constraint_value <= 0.0).all(dim=-1)
        violation = torch.clamp(constraint_value, 0.0).sum(dim=-1)
    else:
        feas = torch.ones(len(f_obj), device=tkwargs["device"], dtype=torch.bool)
        violation = torch.zeros(len(f_obj), **tkwargs)
    return f_obj, feas, violation, dists


def unit_rescale(x: Tensor) -> Tensor:
    """Helper function for normalizing a 1D input to [0, 1]."""
    if not x.dim() == 1:
        raise RuntimeError(f"Expected a 1D input, got shape: {list(x.shape)}")
    if x.min() == x.max():
        return 0.5 * torch.ones(x.shape, dtype=x.dtype, device=x.device)
    return (x - x.min()) / (x.max() - x.min())


def select_based_hvc(predicted_y, new_y, all_num, ref_point):
    predicted_y = -predicted_y.clone().cpu().numpy()
    new_y = -new_y.clone().cpu().numpy()
    all_y = np.concatenate([predicted_y, new_y], axis=0)
    select_y = []
    nds = NonDominatedSorting()
    ranks = nds.do(all_y)
    index = 0

    while len(select_y) < all_num:
        # Choose based on Pareto rank
        if len(ranks[index]) <= all_num - len(select_y):
            select_y.extend(ranks[index])
            index += 1
        else:
            hv = HV(ref_point=ref_point)
            rest_index = []

            for i in range(index, len(ranks)):
                for j in range(len(ranks[i])):
                    rest_index.append(ranks[i][j])
            rest_index = np.array(rest_index)
            rest_y = all_y[rest_index]
            all_hv = hv.calc(rest_y)

            avaliable_num = len(ranks[index])
            while len(select_y) < all_num and avaliable_num > 0:
                max_hvc = -1
                max_hv_index = None

                for i in range(avaliable_num):
                    curr_mask = np.ones(len(rest_y), dtype=bool)
                    curr_mask[i] = False
                    curr_y = rest_y[curr_mask]
                    curr_hv = hv.calc(curr_y)
                    hvc = all_hv - curr_hv
                    if hvc >= max_hvc:
                        max_hvc = hvc
                        max_hv_index = i

                if max_hv_index is not None:
                    select_y.append(rest_index[max_hv_index].item())
                    mask = np.ones(len(rest_index), dtype=bool)
                    mask[max_hv_index] = False
                    rest_y = rest_y[mask]
                    rest_index = rest_index[mask]
                    avaliable_num -= 1
                else:
                    break
    return all_y[select_y], select_y

def Greed_select_psl(trbo_state, psl_candidate):
    n_trs = trbo_state.tr_hparams.n_trust_regions
    x_indice = get_indices_closest_hypercube(trbo_state, psl_candidate)
    for i in range(n_trs):
        tr = trbo_state.trust_regions[i]
        X_normalized  = normalize(psl_candidate, bounds=tr.bounds)
        inds_next_in_tr = get_indices_in_hypercube(
                X_center=tr.X_center_normalized, X=X_normalized, length=tr.length
        )
        num_in_tr = inds_next_in_tr.shape[0]
        if num_in_tr != 0:
            x_indice[inds_next_in_tr] = i

    psl_candidate = unnormalize(X_normalized, bounds = trbo_state.bounds)
    Y_candidate = torch.empty([0,trbo_state.num_objectives]).to('cuda')
    for j in range(len(x_indice)):
        tr_idx = x_indice[j]
        model = trbo_state.models[tr_idx]
        y = model.posterior(psl_candidate[j]).sample(torch.Size([1])).squeeze(0)
        Y_candidate = torch.cat([Y_candidate,y],dim = 0)

    Y_p = trbo_state.pareto_Y_better_than_ref

    ref_point_cpu = trbo_state.ref_point.clone().cpu().numpy()
    _, best_subset = select_based_hvc(Y_p, Y_candidate, 50, ref_point=ref_point_cpu)
    # _,best_subset = select_based_hvc(Y_p,Y_candidate,50,ref_point=trbo_state.ref_point.clone().numpy())
    # best_subset_list = torch.empty([0]).to('cuda').to(torch.long)   
    # for b in range(20):
    #     best_hv_value = 0.
    #     best_subset = None
    #     hv = Hypervolume(ref_point = trbo_state.ref_point.clone())
    #     for k in range(len(Y_candidate)):
    #         Y_subset = Y_candidate[k]
    #         hv_value_without = hv.compute(Y_p)
    #         Y_comb = torch.vstack([Y_p,Y_subset])
    #         hv_value_subset = hv.compute(Y_comb)
    #         hv_contribution = hv_value_subset - hv_value_without
    #         if hv_contribution > best_hv_value:
    #             best_hv_value = hv_contribution
    #             best_subset = torch.Tensor([k]).to('cuda').to(torch.long)
    #     if best_subset is not None:
    #         Y_p = torch.cat((Y_p,Y_candidate[best_subset]),dim = 0)
    #         best_subset_list = torch.cat((best_subset_list,best_subset),dim = -1).to(torch.long)
       
    #print(f'best:{best_subset_list}')
    best_subset_tensor = torch.tensor(best_subset, device=x_indice.device, dtype=torch.long)

    # 过滤掉超出x_indice长度的索引
    valid_indices = best_subset_tensor < x_indice.size(0)

    # 更新best_subset以只包含有效索引
    best_subset_filtered = best_subset_tensor[valid_indices]

    X_next = psl_candidate[best_subset_filtered]

    tr_indices_selected = x_indice[best_subset_filtered]

    return CandidateSelectionOutput(X_cand=X_next, tr_indices=tr_indices_selected)
    
def TS_select_batch_MORBO(trbo_state: TRBOState) -> CandidateSelectionOutput:
    r"""Generate a batch using Thompson sampling as presented in the MORBO.

    Select points using Thompson sampling. When using hypervolume we do greedy selection
    across all trust regions. When using random scalarizations we select a trust region at
    random and then generate a candidate using that trust region as comparing scalarizations
    from different trust regions doesn't work well.

    If there is no feasible candidate we choose the candidate that minimizes the total constraint
    violation. If hypervolume improvement is used but no candidate has non-zero hypervolume improvement
    then we pick the candidate according to a random scalarization.
    """
    tkwargs = {"device": trbo_state.bounds.device, "dtype": trbo_state.bounds.dtype}
    dim = trbo_state.dim
    batch_size = trbo_state.tr_hparams.batch_size
    n_trs = len(trbo_state.trust_regions)
    X_next = torch.empty(0, dim, **tkwargs)
    use_rffs = trbo_state.tr_hparams.use_simple_rff

    # We currently just pick a random trust region to start with and then loop over
    # the trust regions consecutively.
    tr_indices_selected = torch.zeros(
        batch_size, device=tkwargs["device"], dtype=torch.long
    )
    time_sampling, time_hvi = 0, 0
    for i in range(batch_size):
        if trbo_state.tr_hparams.hypervolume:  # Greedy selection
            tr_indices = torch.arange(n_trs, device=tkwargs["device"])
        else:  # Greedy selection doesn't work well for random scalarizations
            tr_indices = torch.randint(
                n_trs, (1,), dtype=torch.long, device=tkwargs["device"]
            )

        # There are three different selection rules depending on constraints etc.
        # (1) Minimize violaton (2) Breaking ties (3) HVI
        best_cand = None
        best_acqval = float("-inf")
        best_tr_idx = -1
        best_selection_rule = -1

        # Loop over all trust regions we are considering
        #如果在center周围有，就是新的best_x,如果没有就是返回中心点本身，这里就存在问题，边界太小就会一直返回中心点本身
        for tr_idx in tr_indices:
            tr = trbo_state.trust_regions[tr_idx]
            if tr.tr_hparams.hypervolume:
                best_X = normalize(tr.best_X, bounds=tr.bounds)
                # only use pareto points inside TR for generating candidates
                indices = get_indices_in_hypercube(
                    tr.X_center_normalized, best_X, length=tr.length
                )
                best_X = best_X[indices]
            else:
                best_X = tr.X_center_normalized

            # Perturbation probability
            prob_perturb = min(64.0 / dim, 1.0) * decay_function(
                n=max(trbo_state.tr_hparams.n_initial_points, trbo_state.n_evals),
                n0=trbo_state.tr_hparams.n_initial_points,
                n_max=max(trbo_state.max_evals, trbo_state.n_evals),
                alpha=0.5,
            ) + random.uniform(0, 0.1)
            #增加了随机扰动
            # Pending points that are in this hypercube
            inds_next_in_tr = get_indices_in_hypercube(
                X_center=tr.X_center_normalized, X=X_next, length=tr.length
            )

            # Sample from all Pareto optimal points in the TR
            X_cand = sample_tr_discrete_points_subset_d(
                best_X=best_X,
                normalized_tr_bounds=tr.get_bounds(),
                n_discrete_points=trbo_state.tr_hparams.raw_samples,
                length=tr.length,
                qmc=trbo_state.tr_hparams.qmc,
                trunc_normal_perturb=trbo_state.tr_hparams.trunc_normal_perturb,
                prob_perturb=prob_perturb,
            )

            # Unnormalize initial conditions to the original hypercube for prediction
            X_cand_unnormalized = unnormalize(X_cand, bounds=tr.bounds)

            objective = trbo_state.trust_regions[tr_idx].objective
            model = trbo_state.models[tr_idx]

            # TODO: Make num_rff_features a hyperparameter of TuRBO
            if use_rffs:
                models = [model] if not isinstance(model, ModelListGP) else model.models
                sample_model = get_gp_samples(
                    model=model,
                    num_outputs=len(models),
                    n_samples=1,
                    num_rff_features=1024,
                )

            # Get the pending points inside the TR and stack them to the candidates
            if len(inds_next_in_tr) > 0:
                X_next_unnormalized = unnormalize(
                    X_next[inds_next_in_tr], bounds=tr.bounds
                )  # Unnormalize pending points for prediction
                X_cand_unnormalized = torch.cat(
                    (X_cand_unnormalized, X_next_unnormalized)
                )

            start = time.time()
            # TODO: Remove the `max_eager_kernel_size` setting when
            # https://github.com/cornellius-gp/gpytorch/issues/1853 has been fixed.
            with torch.no_grad(), gpytorch.settings.fast_computations(
                log_prob=False,
                covar_root_decomposition=False,
                solves=False,
            ), gpytorch.settings.max_eager_kernel_size(float("inf")):
                if use_rffs:
                    Y_sample = (
                        sample_model(X_cand_unnormalized).to(**tkwargs).squeeze(0)
                    )
                else:
                    Y_sample = (
                        model.posterior(X_cand_unnormalized)
                        .sample(torch.Size([1]))
                        .squeeze(0)
                    )
            end = time.time()
            time_sampling += end - start

            # apply objective
            f_obj = objective(Y_sample).clone()

            if trbo_state.constraints is not None:
                constraint_value = torch.stack(
                    [c(Y_sample) for c in trbo_state.constraints], dim=-1
                )
                feas = (constraint_value <= 0.0).all(dim=-1)
                violation = torch.clamp(constraint_value, 0.0).sum(dim=-1)
            else:
                feas = torch.ones(
                    len(f_obj), device=tkwargs["device"], dtype=torch.bool
                )
                violation = torch.zeros(len(f_obj), **tkwargs)

            # Remove the pending points and make sure we don't pick them
            if len(inds_next_in_tr) > 0:
                f_obj_next_in_tr = f_obj[-len(inds_next_in_tr) :].clone()
                feas_in_tr = feas[-len(inds_next_in_tr) :].clone()
                # To make sure these are never picked we set the violation to something large
                feas[-len(inds_next_in_tr) :] = False
                violation[-len(inds_next_in_tr) :] = float("inf")

            start = time.time()
            if not any(feas):  # Ignore the objectives if all are infeasible
                selection_rule = 1
                value_score = -1 * violation
                print(f"{i}) No feasible point, minimizing violation")
            else:
                value_score = float("-inf") * torch.ones(len(f_obj), **tkwargs)
                if trbo_state.tr_hparams.hypervolume:
                    ref_point = trbo_state.ref_point.clone()
                    # This indexes so we have to clone here
                    pareto_Y_better_than_ref = objective(
                        trbo_state.pareto_Y_better_than_ref
                    ).clone()

                    # Include pending points inside this TR when computing the HVI
                    if len(inds_next_in_tr) > 0:
                        f_obj_next_in_tr_better_than_ref = f_obj_next_in_tr[
                            feas_in_tr & (f_obj_next_in_tr > ref_point).all(dim=-1)
                        ]  # Feasible predicted values better than the reference point
                        if len(f_obj_next_in_tr_better_than_ref) > 0:
                            pareto_Y_better_than_ref = torch.cat(
                                (
                                    pareto_Y_better_than_ref,
                                    f_obj_next_in_tr_better_than_ref,
                                ),
                                dim=0,
                            )
                            pareto_Y_better_than_ref = pareto_Y_better_than_ref[
                                is_non_dominated(pareto_Y_better_than_ref)
                            ]

                    # Set points that are either infeasible or not better than the
                    # reference point to have value score zero. If there are no
                    # such points or if no candidate point ends up on the Pareto
                    # frontier we use a random scalarization to break ties.
                    better_than_ref = feas & (f_obj > ref_point).all(dim=-1)
                    if any(better_than_ref):
                        f_obj_better_than_ref = f_obj[better_than_ref]  # m x o
                        # compute box decomposition
                        partitioning = get_partitioning(
                            trbo_state=trbo_state,
                            ref_point=ref_point,
                            Y=pareto_Y_better_than_ref,
                        )
                        # create a deterministic model that returns TS samples that we have
                        # already drawn (with an added dim for q=1). This lets us
                        # batch-evaluate the HVI using samples from the joint posterior
                        # over the discrete set.
                        def get_batched_objective_samples(X):
                            # return a raw_samples x 1 x m-dim tensor of feasible objectives
                            return f_obj_better_than_ref.unsqueeze(1)

                        sampled_model = GenericDeterministicModel(
                            f=get_batched_objective_samples,
                            num_outputs=f_obj_better_than_ref.shape[-1],
                        )
                        acqf = qExpectedHypervolumeImprovement(
                            model=sampled_model,
                            ref_point=ref_point,
                            partitioning=partitioning,
                            sampler=SobolQMCNormalSampler(
                                num_samples=1
                            ),  # dummy sampler
                        )
                        with torch.no_grad():
                            # add a q-batch dimension to compute HVI for each
                            # discrete point alone
                            hvi = acqf(  # dummy input
                                X_cand_unnormalized[better_than_ref].unsqueeze(1)
                            ).to(device=tkwargs["device"])
                        pareto_mask = hvi > 0
                    if any(better_than_ref) and any(pareto_mask):
                        # Hypervolume improvement
                        selection_rule = 3
                        value_score[better_than_ref] = hvi
                    else:
                        selection_rule = 2
                        print(f"{i}) Breaking ties using a random scalarization")
                        weights = sample_simplex(
                            d=trbo_state.num_objectives, n=1, **tkwargs
                        )
                        value_score[feas] = (f_obj[feas] @ weights.t()).squeeze(-1)
                else:  # Random scalarization
                    selection_rule = 2
                    value_score[feas] = f_obj[feas]

            end = time.time()
            time_hvi += end - start

            # Pick the best point
            ind_best = value_score.argmax()
            x_best = X_cand[ind_best, :].unsqueeze(0)

            if selection_rule > best_selection_rule or (
                selection_rule == best_selection_rule
                and value_score.max() > best_acqval
            ):
                best_selection_rule = selection_rule
                best_acqval = value_score.max()
                best_tr_idx = tr_idx
                best_cand = x_best.clone()

        # Save the best candidate
        tr_indices_selected[i] = best_tr_idx
        X_next = torch.cat((X_next, best_cand), dim=0)

    # Unnormalize from [0, 1] to original problem space
    # NOTE: tr.bounds is the same for all TRs, so we can use any of them
    X_next = unnormalize(X=X_next, bounds=tr.bounds)

    print(f"Time spent on sampling: {time_sampling:.1f} seconds")
    print(f"Time spent on HVI computations: {time_hvi:.1f} seconds")
    tr_counts = [(tr_indices_selected == i).sum().cpu().item() for i in range(n_trs)]
    print(f"Number of points selected from each TR: {tr_counts}")
    return CandidateSelectionOutput(X_cand=X_next, tr_indices=tr_indices_selected)


# def TS_select_batch_MORBO(trbo_state: TRBOState) -> CandidateSelectionOutput:
#     tkwargs = {"device": trbo_state.bounds.device, "dtype": trbo_state.bounds.dtype}
#     dim = trbo_state.dim
#     batch_size = trbo_state.tr_hparams.batch_size
#     n_trs = len(trbo_state.trust_regions)
#     X_next = torch.empty(0, dim, **tkwargs)
#     use_rffs = trbo_state.tr_hparams.use_simple_rff
#
#     tr_indices_selected = torch.zeros(batch_size, device=tkwargs["device"], dtype=torch.long)
#     time_sampling, time_hvi = 0, 0
#
#     for i in range(batch_size):
#         if trbo_state.tr_hparams.hypervolume:  # Greedy selection
#             tr_indices = torch.arange(n_trs, device=tkwargs["device"])
#         else:  # Greedy selection doesn't work well for random scalarizations
#             tr_indices = torch.randint(n_trs, (1,), dtype=torch.long, device=tkwargs["device"])
#
#         best_cand = None
#         best_acqval = float("-inf")
#         best_tr_idx = -1
#         best_selection_rule = -1
#
#         for tr_idx in tr_indices:
#             tr = trbo_state.trust_regions[tr_idx]
#             if tr.tr_hparams.hypervolume:
#                 best_X = normalize(tr.best_X, bounds=tr.bounds)
#                 indices = get_indices_in_hypercube(tr.X_center_normalized, best_X, length=tr.length)
#                 best_X = best_X[indices]
#             else:
#                 best_X = tr.X_center_normalized
#
#             prob_perturb = min(64.0 / dim, 1.0) * decay_function(
#                 n=max(trbo_state.tr_hparams.n_initial_points, trbo_state.n_evals),
#                 n0=trbo_state.tr_hparams.n_initial_points,
#                 n_max=max(trbo_state.max_evals, trbo_state.n_evals),
#                 alpha=0.5,
#             )
#
#             inds_next_in_tr = get_indices_in_hypercube(
#                 X_center=tr.X_center_normalized, X=X_next, length=tr.length
#             )
#
#             X_cand = sample_tr_discrete_points_subset_d(
#                 best_X=best_X,
#                 normalized_tr_bounds=tr.get_bounds(),
#                 n_discrete_points=trbo_state.tr_hparams.raw_samples,
#                 length=tr.length,
#                 qmc=trbo_state.tr_hparams.qmc,
#                 trunc_normal_perturb=trbo_state.tr_hparams.trunc_normal_perturb,
#                 prob_perturb=prob_perturb,
#             )
#
#             X_cand_unnormalized = unnormalize(X_cand, bounds=tr.bounds)
#             objective = trbo_state.trust_regions[tr_idx].objective
#             model = trbo_state.models[tr_idx]
#
#             if use_rffs:
#                 models = [model] if not isinstance(model, ModelListGP) else model.models
#                 sample_model = get_gp_samples(
#                     model=model,
#                     num_outputs=len(models),
#                     n_samples=1,
#                     num_rff_features=1024,
#                 )
#
#             if len(inds_next_in_tr) > 0:
#                 X_next_unnormalized = unnormalize(X_next[inds_next_in_tr], bounds=tr.bounds)
#                 X_cand_unnormalized = torch.cat((X_cand_unnormalized, X_next_unnormalized))
#
#             start = time.time()
#             with torch.no_grad(), gpytorch.settings.fast_computations(
#                     log_prob=False,
#                     covar_root_decomposition=False,
#                     solves=False,
#             ), gpytorch.settings.max_eager_kernel_size(float("inf")):
#                 if use_rffs:
#                     Y_sample = (
#                         sample_model(X_cand_unnormalized).to(**tkwargs).squeeze(0)
#                     )
#                 else:
#                     Y_sample = (
#                         model.posterior(X_cand_unnormalized)
#                             .sample(torch.Size([1]))
#                             .squeeze(0)
#                     )
#             end = time.time()
#             time_sampling += end - start
#
#             f_obj = objective(Y_sample).clone()
#             if trbo_state.constraints is not None:
#                 constraint_value = torch.stack(
#                     [c(Y_sample) for c in trbo_state.constraints], dim=-1
#                 )
#                 feas = (constraint_value <= 0.0).all(dim=-1)
#                 violation = torch.clamp(constraint_value, 0.0).sum(dim=-1)
#             else:
#                 feas = torch.ones(
#                     len(f_obj), device=tkwargs["device"], dtype=torch.bool
#                 )
#                 violation = torch.zeros(len(f_obj), **tkwargs)
#
#             if len(inds_next_in_tr) > 0:
#                 f_obj_next_in_tr = f_obj[-len(inds_next_in_tr):].clone()
#                 feas_in_tr = feas[-len(inds_next_in_tr):].clone()
#                 feas[-len(inds_next_in_tr):] = False
#                 violation[-len(inds_next_in_tr):] = float("inf")
#
#             start = time.time()
#             if not any(feas):  # Ignore the objectives if all are infeasible
#                 selection_rule = 1
#                 value_score = -1 * violation
#             else:
#                 if trbo_state.tr_hparams.hypervolume:
#                     ref_point = trbo_state.ref_point.clone()
#                     pareto_Y_better_than_ref = objective(trbo_state.pareto_Y_better_than_ref).clone()
#
#                     if len(inds_next_in_tr) > 0:
#                         f_obj_next_in_tr_better_than_ref = f_obj_next_in_tr[
#                             feas_in_tr & (f_obj_next_in_tr > ref_point).all(dim=-1)
#                             ]
#                         if len(f_obj_next_in_tr_better_than_ref) > 0:
#                             pareto_Y_better_than_ref = torch.cat(
#                                 (pareto_Y_better_than_ref, f_obj_next_in_tr_better_than_ref),
#                                 dim=0,
#                             )
#                             pareto_Y_better_than_ref = pareto_Y_better_than_ref[
#                                 is_non_dominated(pareto_Y_better_than_ref)
#                             ]
#
#                     better_than_ref = feas & (f_obj > ref_point).all(dim=-1)
#                     if any(better_than_ref):
#                         f_obj_better_than_ref = f_obj[better_than_ref]
#                         partitioning = get_partitioning(
#                             trbo_state=trbo_state,
#                             ref_point=ref_point,
#                             Y=pareto_Y_better_than_ref,
#                         )
#
#                         def get_batched_objective_samples(X):
#                             return f_obj_better_than_ref.unsqueeze(1)
#
#                         sampled_model = GenericDeterministicModel(
#                             f=get_batched_objective_samples,
#                             num_outputs=f_obj_better_than_ref.shape[-1],
#                         )
#                         acqf = qExpectedHypervolumeImprovement(
#                             model=sampled_model,
#                             ref_point=ref_point,
#                             partitioning=partitioning,
#                             sampler=SobolQMCNormalSampler(num_samples=1),
#                         )
#                         with torch.no_grad():
#                             hvi = acqf(X_cand_unnormalized[better_than_ref].unsqueeze(1)).to(device=tkwargs["device"])
#                         pareto_mask = hvi > 0
#                     if any(better_than_ref) and any(pareto_mask):
#                         selection_rule = 3
#                         value_score[better_than_ref] = hvi
#                     else:
#                         selection_rule = 2
#                         weights = sample_simplex(d=trbo_state.num_objectives, n=1, **tkwargs)
#                         value_score[feas] = (f_obj[feas] @ weights.t()).squeeze(-1)
#                 else:
#                     selection_rule = 2
#                     value_score[feas] = f_obj[feas]
#             end = time.time()
#             time_hvi += end - start
#
#             # 使用加权和期望改进（EI）策略选择候选点
#             weights = torch.ones(trbo_state.num_objectives) / trbo_state.num_objectives  # 简单的等权重
#             ei_values = weighted_sum_expected_improvement(models, X_cand_unnormalized, weights)
#             ind_best = ei_values.argmax()
#             x_best = X_cand[ind_best, :].unsqueeze(0)
#
#             if selection_rule > best_selection_rule or (
#                     selection_rule == best_selection_rule and value_score.max() > best_acqval
#             ):
#                 best_selection_rule = selection_rule
#                 best_acqval = value_score.max()
#                 best_tr_idx = tr_idx
#                 best_cand = x_best.clone()
#
#         tr_indices_selected[i] = best_tr_idx
#         X_next = torch.cat((X_next, best_cand), dim=0)
#
#     X_next = unnormalize(X=X_next, bounds=tr.bounds)
#
#     print(f"Time spent on sampling: {time_sampling:.1f} seconds")
#     print(f"Time spent on HVI computations: {time_hvi:.1f} seconds")
#     tr_counts = [(tr_indices_selected == i).sum().cpu().item() for i in range(n_trs)]
#     print(f"Number of points selected from each TR: {tr_counts}")
#     return CandidateSelectionOutput(X_cand=X_next, tr_indices=tr_indices_selected)


def weighted_sum_expected_improvement(models, X, weights, xi=0.01):
    mu = sum(w * model.predict(X)[0] for w, model in zip(weights, models))
    sigma = sum(w * model.predict(X)[1] for w, model in zip(weights, models))
    mu_sample_opt = np.max(mu)
    imp = mu - mu_sample_opt - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei