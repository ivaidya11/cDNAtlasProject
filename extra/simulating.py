import numpy as np
import matplotlib.pyplot as plt
import poreflow as pf
from poreflow.classification.predictdna import predict

# ─── Parameters ───────────────────────────────────────────────────────────────

TEMPLATE = 'TTTTTTTTTTTCCTTTTATCGTCATCATCTTTGTAATCGCCGCTGTAGCTGCCATCGCTGCCTTCGCTT'
PEPTIDE  = 'ARND'
IOS      = 240   # pA — open state current
ALPHA    = 20    # pA — volume effect scaling
BETA     = 5     # pA — charge effect scaling
SIGMA    = 8     # pA — Gaussian noise
PEP_BASELINE = 0.5 * IOS  # pA — peptide sits above DNA

AA_PROPERTIES = {
    'A': {'volume': 0.11, 'charge':  0},
    'R': {'volume': 0.63, 'charge': +1},
    'N': {'volume': 0.29, 'charge':  0},
    'D': {'volume': 0.27, 'charge': -1},
}

# ─── DNA region ───────────────────────────────────────────────────────────────

result = predict(TEMPLATE)
dna_steps = result["step_idx"]
dna_pA    = result[pf.MEAN_COL] * IOS

# ─── Linker region ────────────────────────────────────────────────────────────

linker_levels = np.linspace(dna_pA.iloc[-1], PEP_BASELINE, 6)
linker_steps  = np.arange(len(linker_levels)) + dna_steps.max() + 1

# ─── Peptide region ───────────────────────────────────────────────────────────

pep_levels = []
for aa in PEPTIDE:
    level = PEP_BASELINE - ALPHA * AA_PROPERTIES[aa]['volume'] + BETA * AA_PROPERTIES[aa]['charge']
    pep_levels.extend([level, level])  # two half-steps per amino acid

pep_steps = np.arange(len(pep_levels)) + linker_steps[-1] + 1

# ─── Add noise ────────────────────────────────────────────────────────────────

dna_noisy    = dna_pA      + np.random.normal(0, SIGMA, len(dna_pA))
linker_noisy = linker_levels + np.random.normal(0, SIGMA, len(linker_levels))
pep_noisy    = pep_levels  + np.random.normal(0, SIGMA, len(pep_levels))

# ─── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
ax.step(dna_steps,    dna_noisy,    where='mid', linewidth=1.3, label='DNA',     color='steelblue')
ax.step(linker_steps, linker_noisy, where='mid', linewidth=1.3, label='Linker',  color='green')
ax.step(pep_steps,    pep_noisy,    where='mid', linewidth=1.3, label='Peptide', color='orange')
ax.axhline(IOS, color='gray', linestyle='--', label='Open state')
ax.set_xlabel('Step number')
ax.set_ylabel('Current (pA)')
ax.set_title(f'Simulated trace — Peptide: {PEPTIDE}')
ax.legend()
ax.grid(True)
plt.savefig('full_trace.png', dpi=150)
plt.close()
print("done")