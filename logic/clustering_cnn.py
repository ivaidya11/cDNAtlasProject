import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from logic.database import AA_PROPERTIES, IOS, ALPHA, BETA, GAMMA, PEP_BASELINE


amino_acids = list(AA_PROPERTIES.keys())

df_weighted = pd.DataFrame({
    'volume':         ALPHA * pd.Series({aa: AA_PROPERTIES[aa]['volume'] for aa in amino_acids}),
    'charge':         BETA  * pd.Series({aa: AA_PROPERTIES[aa]['charge'] for aa in amino_acids}),
    'hydrophobicity': GAMMA * pd.Series({aa: AA_PROPERTIES[aa]['hydrophobicity'] for aa in amino_acids}),
}, index=amino_acids)

df_weighted['score'] = (PEP_BASELINE * IOS
                        - df_weighted['volume']
                        - df_weighted['charge']
                        - df_weighted['hydrophobicity'])

linkage_matrix = linkage(df_weighted['score'].values.reshape(-1, 1), method='ward')


def get_clusters(n_clusters):
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    return {aa: int(labels[i]) for i, aa in enumerate(amino_acids)}


def plot_dendrogram():
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(linkage_matrix, labels=amino_acids, ax=ax)
    ax.set_title('Hierarchical clustering in current space (score)')
    ax.set_xlabel('Amino acid')
    ax.set_ylabel('Current distance (pA)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_dendrogram()