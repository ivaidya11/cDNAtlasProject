"""
HMM segmentation logic for nanopore traces.
Import this module from the notebook; all plotting stays in the notebook.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGION_TO_STATE = {'DNA': 0, 'linker': 1, 'peptide': 1}
STATE_TO_REGION = {0: 'DNA', 1: 'non-DNA'}
REGION_COLORS   = {'DNA': '#4895ef', 'non-DNA': '#f72585'}

TRANSMAT   = np.array([
    [0.9999, 0.0001],  # from DNA
    [0.0000, 1.0000],  # from non-DNA
])
STARTPROB  = np.array([1.0, 0.0])

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def fit_emission_params(train_db: pd.DataFrame) -> tuple[list, list]:
    """Compute per-state Gaussian emission mean/std from labelled training data."""
    dna_samples    = train_db[train_db['region'] == 'DNA']['current_pA']
    nondna_samples = train_db[
        (train_db['region'] == 'peptide') | (train_db['region'] == 'linker')
    ]['current_pA']

    means = [dna_samples.mean(), nondna_samples.mean()]
    stds  = [dna_samples.std(),  nondna_samples.std()]
    return means, stds


def build_model(train_db: pd.DataFrame) -> GaussianHMM:
    """Build and return the GaussianHMM with fixed parameters from training data."""
    means, stds = fit_emission_params(train_db)

    means_arr  = np.array(means).reshape(2, 1)
    covars_arr = np.array(stds).reshape(2, 1, 1) ** 2

    model = GaussianHMM(n_components=2, covariance_type='full', n_iter=0)
    model.n_features = 1
    model.startprob_ = STARTPROB.astype(float)
    model.transmat_  = TRANSMAT
    model.means_     = means_arr
    model.covars_    = covars_arr
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_predictions(
    model: GaussianHMM,
    test_db: pd.DataFrame,
) -> tuple[dict, dict]:
    """
    Run Viterbi on every trace in test_db.

    Returns
    -------
    accuracies  : {trace_id: float}
    predictions : {trace_id: dict with keys time_ms, current_pA, true_states, pred_states}
    """
    accuracies:  dict = {}
    predictions: dict = {}

    for tid in test_db['trace_id'].unique():
        trace = test_db[test_db['trace_id'] == tid]
        X = trace['current_pA'].values.reshape(-1, 1)

        pred_states = model.predict(X)
        true_states = trace['region'].map(REGION_TO_STATE).values

        accuracies[tid] = float(np.mean(pred_states == true_states))
        predictions[tid] = {
            'time_ms':     trace['time_ms'].values,
            'current_pA':  trace['current_pA'].values,
            'true_states': true_states,
            'pred_states': pred_states,
        }

    return accuracies, predictions


def compute_boundary_errors(predictions: dict) -> dict:
    """
    Compute peptide-start boundary error (ms) for each trace.

    Positive value  → predicted late
    Negative value  → predicted early
    """
    PEPTIDE_STATE = REGION_TO_STATE['peptide']
    boundary_errors: dict = {}

    for tid, data in predictions.items():
        true_idxs = np.where(data['true_states'] == PEPTIDE_STATE)[0]
        pred_idxs = np.where(data['pred_states'] == PEPTIDE_STATE)[0]

        if len(true_idxs) == 0 or len(pred_idxs) == 0:
            continue

        boundary_errors[tid] = (
            float(data['time_ms'][pred_idxs[0]]) - float(data['time_ms'][true_idxs[0]])
        )

    return boundary_errors


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def extract_peptide_sections(predictions: dict) -> dict:
    """
    For each trace, extract the raw samples that the HMM labelled as non-DNA (peptide).

    Returns
    -------
    {trace_id: {'time_ms': np.ndarray, 'current_pA': np.ndarray}}
        Only samples where pred_states == REGION_TO_STATE['peptide'].
    """
    PEPTIDE_STATE = REGION_TO_STATE['peptide']
    sections: dict = {}

    for tid, data in predictions.items():
        mask = data['pred_states'] == PEPTIDE_STATE
        if not mask.any():
            continue
        sections[tid] = {
            'time_ms':    data['time_ms'][mask],
            'current_pA': data['current_pA'][mask],
        }

    return sections


# ---------------------------------------------------------------------------
# Plotting helper (shared between notebook and any scripts)
# ---------------------------------------------------------------------------

def shade_regions(ax, t, states, ylo, yhi, alpha: float = 0.85) -> None:
    """Fill colour bands on *ax* based on HMM state labels."""
    for state, region in STATE_TO_REGION.items():
        mask = states == state
        if mask.any():
            ax.fill_between(
                t, ylo, yhi,
                where=mask,
                color=REGION_COLORS[region],
                alpha=alpha,
                linewidth=0,
            )
