# extract certain features of each step int he trace, such as mean, standard deviation, dwell time,
# and have the label as the amino acid sequence

# first change database.py to store the peptide sequence
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from logic.database import AA_CLASS



db = pd.read_csv('../nanopore_trace_database.csv')

features = []

for trace_id, trace in db.groupby('trace_id'):
    # keep only the peptide region
    peptide = trace[trace['region'] == 'peptide'].reset_index(drop=True)
    
    # Assign step IDs by detecting where clean_pA changes
    step_ids = (peptide['clean_pA'] != peptide['clean_pA'].shift()).cumsum()
    peptide = peptide.copy()
    peptide['step_id'] = step_ids
    
    peptide_seq = peptide['peptide_sequence'].iloc[0]
    aa_list = list(peptide_seq)
    
    for step_id, step in peptide.groupby('step_id'):
        # step_id is 1-indexed, map to amino acid
        aa_idx = step_id - 1
        if aa_idx >= len(aa_list):
            continue
            
        features.append({
            'trace_id': trace_id,
            'step_id': step_id,
            'mean_current': step['current_pA'].mean(),
            'std_current': step['current_pA'].std(),
            'dwell_time': step['time_ms'].max() - step['time_ms'].min(),
            'amino_acid': aa_list[aa_idx]
        })


df_features = pd.DataFrame(features)
df_features['phys_class'] = [AA_CLASS.get(aa, 'Unknown') for aa in df_features['amino_acid']]

print(df_features.head(20))
df_features.to_csv('features.csv', index=False)

print(f"\nTotal steps: {len(df_features)}")
print(f"\nAmino acid distribution:\n{df_features['amino_acid'].value_counts()}")
