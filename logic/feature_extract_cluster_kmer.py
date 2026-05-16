import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import seaborn as sns
import pandas as pd
from logic.database import WINDOW_HALF
# from logic.feature_extract_clusters import CLUSTERS
from logic.clustering_cnn import get_clusters


# df_clusters = pd.read_csv('data/amino_acid_clusters.csv')
db = pd.read_csv('data/randomcontrol_trace_database.csv')


final_save_path = 'data/random_control_feature_by_kmer_cluster.csv'
clusters = get_clusters(n_clusters=7)
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
        aa_minus2 = aa_list[aa_idx - 2] if aa_idx - 2 >= 0 else None   
        step_minus1 = step_groups.get(sid - 1)
        aa_minus1 = aa_list[aa_idx - 1] if aa_idx - 1 >= 0 else None
        step_plus1 = step_groups.get(sid + 1)
        aa_plus1 = aa_list[aa_idx + 1] if aa_idx + 1 < len(aa_list) else None
        step_plus2 = step_groups.get(sid + 2)
        aa_plus2 = aa_list[aa_idx + 2] if aa_idx + 2 < len(aa_list) else None
        ratio = []
        for i in range(-WINDOW_HALF, WINDOW_HALF+1):
            weight_formula = np.exp(-i ** 2 / 2.0)
            ratio.append(weight_formula)

        def cluster_of(aa):
            return str(clusters[aa]) if aa is not None else 'X'

        cluster_kmer = '_'.join([
            cluster_of(aa_minus2),
            cluster_of(aa_minus1),
            cluster_of(aa_list[aa_idx]),
            cluster_of(aa_plus1),
            cluster_of(aa_plus2),
        ])

        features.append({
            'trace_id': trace_id,
            'step_id': sid,
            'mean_current': grp['current_pA'].mean(),
            'std_current': grp['current_pA'].std(),
            'dwell_time': grp['time_ms'].max() - grp['time_ms'].min(),

            'mean_minus2': ratio[0]* step_minus2['current_pA'].mean() if step_minus2 is not None else np.nan,

            'mean_minus1': ratio[1]* step_minus1['current_pA'].mean() if step_minus1 is not None else np.nan,
            
            'mean_plus1': ratio[3]* step_plus1['current_pA'].mean() if step_plus1 is not None else np.nan,
            
            'mean_plus2': ratio[4]* step_plus2['current_pA'].mean() if step_plus2 is not None else np.nan,

            'cluster_kmer': cluster_kmer
        })


df_features = pd.DataFrame(features)

print(df_features.head(20))
df_features.to_csv(final_save_path, index=False)

print(f"\nTotal steps: {len(df_features)}")
print(f"\nAmino acid distribution:\n{df_features['cluster_kmer'].value_counts()}")


# print(peptide.info())

