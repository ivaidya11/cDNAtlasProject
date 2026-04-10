import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import pandas as pd
from database import AA_PROPERTIES, AA_CLASS
# from onelettercodes import aa_codes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist



amino_acids = list(AA_PROPERTIES.keys())

df = pd.DataFrame({
    'volume': [AA_PROPERTIES[aa]['volume'] for aa in amino_acids],
    'hydrophobicity': [AA_PROPERTIES[aa]['hydrophobicity'] for aa in amino_acids],
    'charge': [AA_PROPERTIES[aa]['charge'] for aa in amino_acids],
}, index=amino_acids)


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.values)
df_scaled = pd.DataFrame(X_scaled, index=amino_acids, columns=df.columns)

# 'ward' minimizes variance within clusters — good default
linkage_matrix = linkage(X_scaled, method='ward')

fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(
    linkage_matrix,
    labels=amino_acids,
    ax=ax,
    color_threshold=3.5  # adjust this to change where clusters are cut
)
ax.set_title('Hierarchical clustering of amino acids by simulation properties')
ax.set_xlabel('Amino acid')
ax.set_ylabel('Distance (Ward)')
ax.axhline(y=3.5, color='red', linestyle='--', label='Cut threshold')
ax.legend()
plt.tight_layout()
plt.savefig('extra/amino_acid_dendrogram.png')

n_clusters = 4  # start with 4 to compare against your physicochemical scheme
labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

cluster_df = pd.DataFrame({'AA': amino_acids, 'cluster': labels})
for c in sorted(cluster_df['cluster'].unique()):
    members = cluster_df[cluster_df['cluster'] == c]['AA'].tolist()


cluster_df.to_csv('data/amino_acid_clusters.csv', index=False)