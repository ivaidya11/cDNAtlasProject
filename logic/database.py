"""

Generates synthetic ionic current traces for DNA-peptide constructs
translocating through MspA

"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import poreflow as pf
from poreflow.steps import predict

def linear_interpolation(target_min, target_max, scale_min, scale_max, dict):
    scaled_dict = {}
    for key, value in dict.items():
        scaled_value = (((value - scale_min) * (target_max - target_min)) / (scale_max - scale_min)) + target_min
        scaled_dict[key] = scaled_value
    return scaled_dict

# ─── Simulation Parameters ────────────────────────────────────────────────────

IOS             = 240   # pA — open state current (lowered from 320 to match real data)
PEP_BASELINE    = 0.5  # fraction of IOS for peptide baseline (~132 pA)
ALPHA           = 80   # pA — how much volume affects current (bigger AA = more blockage) #used to be 20, changed alpha and beta so that peptide has a larger effect on the current
BETA            = 20   # pA — how much charge affects current per unit charge #used to be 5
GAMMA          = 15   # pA — how much hydrophobicity affects current per unit (0-1) hydrophobicity
SIGMA_WHITE     = 1.0   # pA — white (Gaussian) noise standard deviation
SIGMA_FLICKER   = 1.0   # pA — 1/f (flicker) noise amplitude
N_TRACES        = 10000  # number of traces to generate
PEP_LENGTH      = 20     # amino acids per peptide # increased from 8 to 20
LINKER_STEPS    = 6     # number of levels in linker region


# Multi-residue sensing window (MspA constriction zone spans ~3-5 AAs)
WINDOW_HALF     = 2  # residues on each side of center (total window = 2*WINDOW_HALF+1 = 5) - 
#changed it from 3 to 1 to get a wider peptide current so this might be inaccurate now

# Timeseries parameters
SAMPLING_FREQ   = 5000  # Hz — samples per second
MEAN_DWELL_MS   = 5   # ms — mean dwell time per step/residue (exponential distribution)

# ─── Amino Acid Properties ────────────────────────────────────────────────────
# Volume: normalized van der Waals volume (0 = smallest, 1 = largest)
# Charge: formal charge at neutral pH
#
# IMPORTANT — charge direction convention (Motone et al. 2024):
#   Positive charge (K, R) → DECREASES current (attracts Cl-, adds volume)
#   Negative charge (D, E) → INCREASES current (resists translocation, stretches strand)
#   This is the OPPOSITE of naive intuition — see beta term in level calculation below

AA_PROPERTIES = {
    'G': {'charge':  0},  # Glycine
    'A': {'charge':  0},  # Alanine
    'S': {'charge':  0},  # Serine
    'T': {'charge':  0},  # Threonine
    'C': {'charge':  0},  # Cysteine
    'N': {'charge':  0},  # Asparagine
    'D': {'charge': -1},  # Aspartate     — negative charge
    'E': {'charge': -1},  # Glutamate     — negative charge
    'Q': {'charge':  0},  # Glutamine
    'V': {'charge':  0},  # Valine
    'L': {'charge':  0},  # Leucine
    'I': {'charge':  0},  # Isoleucine
    'P': {'charge':  0},  # Proline
    'M': {'charge':  0},  # Methionine
    'H': {'charge':  0},  # Histidine
    'K': {'charge': +1},  # Lysine        — positive charge
    'R': {'charge': +1},  # Arginine      — positive, large
    'F': {'charge':  0},  # Phenylalanine
    'Y': {'charge':  0},  # Tyrosine
    'W': {'charge':  0},  # Tryptophan
}

# Raw van der Waals volumes (Å³) — replace these with paper values
VOLUME_OG = {
    'G':  34.8,  # Glycine       — smallest
    'A':  52.0,  # Alanine       — small, neutral
    'S':  51.9,  # Serine        — small, polar
    'T':  68.4,  # Threonine     — polar
    'C':  65.0,  # Cysteine      — small, unique
    'N': 69.6,  # Asparagine    — polar
    'D': 65.4,  # Aspartate     — negative charge
    'E': 77.5,  # Glutamate     — negative charge
    'Q': 85.5,  # Glutamine     — polar
    'V': 82.4,  # Valine        — medium, nonpolar
    'L': 99.1,  # Leucine       — medium, nonpolar
    'I': 99.1,  # Isoleucine    — medium, nonpolar
    'P': 72.6,  # Proline       — rigid
    'M': 96.9,  # Methionine    — medium
    'H': 90.4,  # Histidine     — weakly positive
    'K': 100.1,  # Lysine        — positive charge
    'R': 118.9,  # Arginine      — positive, large
    'F': 112.8,  # Phenylalanine — bulky, aromatic
    'Y': 114.7,  # Tyrosine      — bulky, aromatic
    'W': 135.5,  # Tryptophan    — largest
}

VOLUME = linear_interpolation(0, 1, min(VOLUME_OG.values()), max(VOLUME_OG.values()), VOLUME_OG)
for aa, val in VOLUME.items():
    AA_PROPERTIES[aa]['volume'] = val

AA_LIST = list(AA_PROPERTIES.keys())

AA_CLASS = {
    'K': 'positive', 'R': 'positive',
    'D': 'negative', 'E': 'negative',
    'S': 'polar',    'T': 'polar',    'C': 'polar',
    'N': 'polar',    'Q': 'polar',    'Y': 'polar',    'H': 'polar',
    'G': 'nonpolar', 'A': 'nonpolar', 'V': 'nonpolar', 'L': 'nonpolar',
    'I': 'nonpolar', 'P': 'nonpolar', 'F': 'nonpolar', 'M': 'nonpolar',
    'W': 'nonpolar',
}

HYDROPHOBICITY_OG = {'A': 1.800,
                'R': -4.500,
                'N': -3.500,
                'D': -3.500,
                'C':  2.500,
                'Q': -3.500,
                'E': -3.500,
                'G': -0.400,
                'H': -3.200,
                'I':  4.500,
                'L':  3.800,
                'K': -3.900,
                'M':  1.900,
                'F':  2.800,
                'P': -1.600,
                'S': -0.800,
                'T': -0.700,
                'W': -0.900,
                'Y': -1.300,
                'V':  4.200}

HYDROPHICITY = linear_interpolation(0, 1, -4.5, 4.5, HYDROPHOBICITY_OG)
for aa, val in HYDROPHICITY.items():
    AA_PROPERTIES[aa]['hydrophobicity'] = val

# ─── Codon Table (one codon per AA for simplicity) ───────────────────────────

CODON_TABLE = {
    'G': 'GGT', 'A': 'GCT', 'S': 'TCT', 'T': 'ACT', 'C': 'TGT',
    'N': 'AAT', 'D': 'GAT', 'E': 'GAA', 'Q': 'CAA', 'V': 'GTT',
    'L': 'CTT', 'I': 'ATT', 'P': 'CCT', 'M': 'ATG', 'H': 'CAT',
    'K': 'AAA', 'R': 'CGT', 'F': 'TTT', 'Y': 'TAT', 'W': 'TGG',
}



# ─── Helper Functions ─────────────────────────────────────────────────────────

def peptide_to_cdna(peptide: str) -> str:
    """Convert peptide sequence to cDNA using one codon per amino acid."""
    return ''.join(CODON_TABLE[aa] for aa in peptide)

def _dwell_samples() -> int:
    """Sample a dwell duration: max(5 ms, Exp(MEAN_DWELL_MS)), converted to samples."""
    dwell_ms = max(5.0, np.random.exponential(MEAN_DWELL_MS))
    return int(dwell_ms * SAMPLING_FREQ / 1000)


def simulate_dna_region(cdna: str) -> np.ndarray:
    """
    Predict DNA current levels from cDNA sequence using the 6-mer pore model.
    Each step is expanded into raw samples with an exponentially-distributed dwell.

    Returns current_pA as a 1D array of raw samples.
    """
    result = predict.predict(cdna)
    levels = result[pf.MEAN_COL].values * IOS

    samples = []
    for level in levels:
        samples.extend([level] * _dwell_samples())
    return np.array(samples)


def simulate_linker_region(dna_end: float, pep_start: float) -> np.ndarray:
    """
    Simulate the linker as a short transition between DNA and peptide regions.
    Each interpolated level is expanded into raw samples with an exponentially-distributed dwell.

    Returns current_pA as a 1D array of raw samples.
    """
    levels = np.linspace(dna_end, pep_start, LINKER_STEPS)

    samples = []
    for level in levels:
        samples.extend([level] * _dwell_samples())
    return np.array(samples)


def _compute_window_current(peptide: str, center_idx: int) -> float:
    """
    Compute the current level for one sensing window position.

    Uses a Gaussian-weighted average over WINDOW_HALF residues on each side
    of center_idx, reflecting the MspA constriction zone integrating over
    ~3-5 amino acids simultaneously. Edge positions are clamped (boundary
    residue repeated) rather than zero-padded.

    Three physicochemical properties contribute:
        - Volume:         larger AA → more blockage (ALPHA term)
        - Charge:         positive → decreases current, negative → increases (BETA term)
        - Hydrophobicity: higher hydrophobicity → more blockage (GAMMA term)
          Values are normalised Kyte-Doolittle scores [0,1].

    Returns current in pA.
    """
    pep_baseline = PEP_BASELINE * IOS
    n = len(peptide)

    weighted_volume        = 0.0
    weighted_charge        = 0.0
    weighted_hydrophobicity = 0.0
    total_weight           = 0.0

    for d in range(-WINDOW_HALF, WINDOW_HALF + 1):
        idx = max(0, min(n - 1, center_idx + d))
        aa  = peptide[idx]
        weight = np.exp(-d ** 2 / 2.0)

        weighted_volume         += weight * AA_PROPERTIES[aa]['volume']
        weighted_charge         += weight * AA_PROPERTIES[aa]['charge']
        weighted_hydrophobicity += weight * AA_PROPERTIES[aa]['hydrophobicity']
        total_weight            += weight

    weighted_volume         /= total_weight
    weighted_charge         /= total_weight
    weighted_hydrophobicity /= total_weight

    return (pep_baseline
            - ALPHA * weighted_volume
            - BETA  * weighted_charge
            - GAMMA * weighted_hydrophobicity)

def simulate_peptide_region(peptide: str) -> np.ndarray:
    """
    Simulate peptide current levels using a Gaussian-windowed volume + charge model
    with exponentially-distributed dwell times per amino acid.

    Sensing window (_compute_window_current):
        Each position blends contributions from WINDOW_HALF neighbouring residues
        on each side, weighted by a Gaussian. Reflects MspA's ~3-5 AA constriction.

    Dwell times:
        Drawn from max(5 ms, Exp(MEAN_DWELL_MS)), converted to samples at SAMPLING_FREQ.

    Returns current_pA as a 1D array of raw samples.
    """
    samples = []
    for i in range(len(peptide)):
        level = _compute_window_current(peptide, i)
        samples.extend([level] * _dwell_samples())
    return np.array(samples)


def add_noise(current: np.ndarray, sigma: float = SIGMA_WHITE) -> np.ndarray:
    """
    Add white (Gaussian) + 1/f (flicker) noise to a current trace.

    White noise: independent Gaussian samples with std = sigma.
    Flicker noise: 1/f power spectrum generated via Fourier method, scaled to
        std = SIGMA_FLICKER. Captures the low-frequency drift seen in real
        nanopore recordings.
    """
    n = len(current)

    # White noise
    white = np.random.normal(0, sigma, n)

    # 1/f flicker noise via Fourier method
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid division by zero at DC component
    amplitude = np.sqrt(1.0 / freqs)
    phases = np.random.uniform(0, 2 * np.pi, len(freqs))
    flicker = np.fft.irfft(amplitude * np.exp(1j * phases), n)
    if flicker.std() > 0:
        flicker = flicker * (SIGMA_FLICKER / flicker.std())

    return current + white + flicker


def simulate_trace(peptide: str) -> pd.DataFrame:
    """
    Simulate a full ionic current trace for a given peptide sequence.

    Trace structure:
        [DNA] → [linker] → [peptide]

    Each step is expanded into raw samples at SAMPLING_FREQ Hz with
    exponentially-distributed dwell times (min 5 ms, mean MEAN_DWELL_MS).

    Returns a DataFrame with columns:
        time_ms       — cumulative time in milliseconds (x axis)
        current_pA    — simulated current with noise (y axis)
        region        — ground truth label: 'DNA', 'linker', 'peptide'
    """
    cdna = peptide_to_cdna(peptide)

    # 1. DNA region — helicase ratchets cDNA through pore
    dna_current = simulate_dna_region(cdna)

    # 2. Linker region — junction between DNA and peptide
    linker_current = simulate_linker_region(
        dna_end=dna_current[-1],
        pep_start=PEP_BASELINE * IOS,
    )

    # 3. Peptide region — amino acids pass through sensing region
    pep_current = simulate_peptide_region(peptide)

    # Apply noise per region
    dna_noisy     = add_noise(dna_current)
    linker_noisy  = add_noise(linker_current)
    pep_noisy     = add_noise(pep_current)

    # Concatenate and build time axis
    all_current       = np.concatenate([dna_noisy, linker_noisy, pep_noisy])
    all_clean_current = np.concatenate([dna_current, linker_current, pep_current])
    regions = (
        ['DNA']    * len(dna_current) +
        ['linker'] * len(linker_current) +
        ['peptide'] * len(pep_current)
    )
    time_ms = np.arange(len(all_current)) * (1000 / SAMPLING_FREQ)

    return pd.DataFrame({'time_ms': time_ms, 'current_pA': all_current,
                         'clean_pA': all_clean_current, 'region': regions})


# ─── Generate Database ────────────────────────────────────────────────────────

def generate_database(n_traces: int, pep_length: int, output_path: str) -> pd.DataFrame:
    """
    Generate a database of simulated traces with random peptide sequences.
    Each trace gets a unique trace_id and the peptide sequence is stored
    alongside every row so you can look up what was in the pore.
    """
    all_traces = []

    for trace_id in range(n_traces):
        peptide = ''.join(random.choices(AA_LIST, k=pep_length))

        trace_df = simulate_trace(peptide)
        trace_df['trace_id'] = trace_id
        trace_df['peptide_sequence'] = peptide

        all_traces.append(trace_df)

        if (trace_id + 1) % 1000 == 0:
            print(f"Generated {trace_id + 1} / {n_traces} traces")

    database = pd.concat(all_traces, ignore_index=True)
    database.to_csv(output_path, index=False)

    print(f"\nDatabase saved to: {output_path}")
    print(f"Total rows:  {len(database)}")
    print(f"Columns:     {list(database.columns)}")
    return database


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_example_trace(database: pd.DataFrame, trace_id: int = 0):
    """
    Plot a single trace from the database, coloured by region.
    Use this as a sanity check after generating the database.
    """
    trace = database[database['trace_id'] == trace_id]
    peptide = trace['peptide_sequence'].iloc[0]

    colors = {
        'DNA':        'steelblue',
        'linker':     'green',
        'peptide':    'orange'
    }

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(trace['time_ms'], trace['current_pA'], color='black', linewidth=0.8, zorder=1)
    for region, group in trace.groupby('region', sort=False):
        ax.plot(group['time_ms'], group['current_pA'],
                label=region, color=colors[region], linewidth=0.8, zorder=2)

    ax.axhline(IOS, color='gray', linestyle='--', alpha=0.4, label='IOS reference')
    ax.set_title(f'Trace {trace_id} — Peptide: {peptide}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (pA)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('data/example_trace.png', dpi=150)
    plt.close()
    print("Saved to data/example_trace.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output_path = "data/nanopore_trace_database.csv"

    print(f"Generating {N_TRACES} traces (peptide length = {PEP_LENGTH})...")
    print(f"IOS = {IOS} pA | alpha = {ALPHA} | beta = {BETA} | sigma_white = {SIGMA_WHITE} | sigma_flicker = {SIGMA_FLICKER}\n")

    db = generate_database(N_TRACES, PEP_LENGTH, output_path)
    plot_example_trace(db, trace_id=500)

    print("\n--- Current (pA) by region ---")
    print(db.groupby('region')['current_pA'].describe().round(2))




    