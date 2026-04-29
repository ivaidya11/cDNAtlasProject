import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import seaborn as sns
import pandas as pd
from logic.database import WINDOW_HALF


df_clusters = pd.read_csv('data/amino_acid_clusters.csv')
db = pd.read_csv('data/nanopore_trace_database.csv')


CLUSTERS = {}

for _, row in df_clusters.iterrows():
    aa = row['AA']
    cluster = row['cluster']
    if cluster == 1:
        CLUSTERS[aa] = '1, other'
    elif cluster == 2:
        CLUSTERS[aa] = '2, negative'
    elif cluster == 3:
        CLUSTERS[aa] = '3, large and hydrophobic'
    elif cluster == 4:
        CLUSTERS[aa] = '4, small'

final_save_path = 'data/nanopore_feature_by_cluster.csv'

features = []

for trace_id, trace in db.groupby('trace_id'):
    # keep only the peptide region
    peptide = trace[trace['region'] == 'peptide'].reset_index(drop=True)
    
    # Assign step IDs by detecting where clean_pA changes
    step_ids = (peptide['clean_pA'] != peptide['clean_pA'].shift()).cumsum()
    peptide = peptide.copy()
    peptide['step_id'] = step_ids
    
    peptide_seq = peptide['peptide_sequence'].iloc[0] # arbitrary we're jsut trying to get one of the sequences right?
    aa_list = list(peptide_seq)
##  this is what i need to fix to create a feature vector that contains all the amino acids in the 5-mer window

    step_groups = {sid: grp for sid, grp in peptide.groupby('step_id')}


    for sid, grp in step_groups.items():
        # step_id is 1-indexed, map to amino acid

        aa_idx = sid - 1
        if aa_idx >= len(aa_list):
            continue
        step_minus2 = step_groups.get(sid - 2)
        step_minus1 = step_groups.get(sid - 1)
        step_plus1 = step_groups.get(sid + 1)
        step_plus2 = step_groups.get(sid + 2)
        ratio = []
        for i in range(-WINDOW_HALF, WINDOW_HALF+1):
            weight_formula = np.exp(-i ** 2 / 2.0)
            ratio.append(weight_formula)


        

        features.append({
            'trace_id': trace_id,
            'step_id': sid,
            'mean_current': grp['current_pA'].mean(),
            'std_current': grp['current_pA'].std(),
            'dwell_time': grp['time_ms'].max() - grp['time_ms'].min(),
            'amino_acid': aa_list[aa_idx], # this is the label that the classifer can use to train

            # these are all to use to put into the feature vector for the classifer to train on,
            # i exclude the labels here because the knowledge of what exact amino acid is surrounding
            # the amino acid we're trying to classify is not somethign we'll know with the experiemental data? im still a bit confused on this
            'weighted_window_current': (
                ratio[0] * (step_minus2['current_pA'].mean() if step_minus2 is not None else np.nan) +
                ratio[1] * (step_minus1['current_pA'].mean() if step_minus1 is not None else np.nan) +
                ratio[2] * grp['current_pA'].mean() +
                ratio[3] * (step_plus1['current_pA'].mean() if step_plus1 is not None else np.nan) +
                ratio[4] * (step_plus2['current_pA'].mean() if step_plus2 is not None else np.nan)
            ),
            'mean_minus2': step_minus2['current_pA'].mean() if step_minus2 is not None else np.nan,
            # 'std_minus2': step_minus2['current_pA'].std() if step_minus2 is not None else np.nan,
            # 'dwell_minus2': step_minus2['time_ms'].max() - step_minus2['time_ms'].min() if step_minus2 is not None else np.nan,
            'mean_minus1': step_minus1['current_pA'].mean() if step_minus1 is not None else np.nan,
            # 'std_minus1': step_minus1['current_pA'].std() if step_minus1 is not None else np.nan,
            # 'dwell_minus1': step_minus1['time_ms'].max() - step_minus1['time_ms'].min() if step_minus1 is not None else np.nan,
            'mean_plus1': step_plus1['current_pA'].mean() if step_plus1 is not None else np.nan,
            # 'std_plus1': step_plus1['current_pA'].std() if step_plus1 is not None else np.nan,
            # 'dwell_plus1': step_plus1['time_ms'].max() - step_plus1['time_ms'].min() if step_plus1 is not None else np.nan,
            'mean_plus2': step_plus2['current_pA'].mean() if step_plus2 is not None else np.nan,
            # 'std_plus2': step_plus2['current_pA'].std() if step_plus2 is not None else np.nan,
            # 'dwell_plus2': step_plus2['time_ms'].max() - step_plus2['time_ms'].min() if step_plus2 is not None else np.nan,
            # so here we want a list of amino acis instead of just the one thats associated?
            # we want extra feature columns with data for the previous and the future columsn
        })


df_features = pd.DataFrame(features)
df_features['cluster_class'] = [CLUSTERS.get(aa, 'Unknown') for aa in df_features['amino_acid']]

print(df_features.head(20))
df_features.to_csv(final_save_path, index=False)

print(f"\nTotal steps: {len(df_features)}")
print(f"\nAmino acid distribution:\n{df_features['amino_acid'].value_counts()}")


# print(peptide.info())

