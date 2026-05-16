import pandas as pd
import numpy as np

# takes the feature extracted csv, fits a gaussian emission per unique kmer, builds the shift-by-one transition matrix, and runs viterbi


# emission model
# group training data by cluster_kmer, compute mea and std of mean current per kmer
# that results in the gaussina per state
features = pd.read_csv('data/nanopore_training_BIG_feature_by_kmer_cluster.csv')

def fit_emission_model(feature_csv):
    grouped = feature_csv.groupby('cluster_kmer')['mean_current'].agg(['mean', 'std'])

    return grouped.to_dict('index') # converts it to a nested dict keyed by kmer, easier to look up later

from collections import defaultdict
def transition_matrix(feature_csv):
    # for each consecutive pair of steps in the same trace
    counts = defaultdict(lambda: defaultdict(int))
    feature_kmers = feature_csv.groupby('trace_id')
    for trace_id, trace in feature_kmers:
        kmers = trace.sort_values('step_id')['cluster_kmer'].tolist()
        for (kmer_now, kmer_next) in zip(kmers[:-1], kmers[1:]):
            counts[kmer_now][kmer_next] += 1

    # convert to log probs
    transitions = {}
    for from_state, to_counts in counts.items():
        total = sum(to_counts.values())
        transitions[from_state] = {to: np.log(count/total) for to, count in to_counts.items()}

    return transitions
def gaussian_log_prob(x, mean, std):
    return -0.5 * ((x - mean) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi))


def viterbi(observations, emissions, transitions):
    states = list(emissions.keys())

    # t=0: no transition, just emission
    vit = [{
        state: gaussian_log_prob(observations[0], params['mean'], params['std'])
        for state, params in emissions.items()
    }]
    backpointers = []

    for t in range(1, len(observations)):
        vit_t = {s: -np.inf for s in states}
        bp_t  = {s: None    for s in states}
        for p, to_dict in transitions.items():
            prev_score = vit[t-1][p]
            if prev_score == -np.inf:
                continue
            for s, log_trans in to_dict.items():
                if s not in emissions:
                    continue
                emit  = gaussian_log_prob(observations[t], emissions[s]['mean'], emissions[s]['std'])
                score = prev_score + log_trans + emit
                if score > vit_t[s]:
                    vit_t[s] = score
                    bp_t[s]  = p

        vit.append(vit_t)
        backpointers.append(bp_t)

    # traceback: start from best state at last timestep
    best_last = max(vit[-1], key=vit[-1].get)
    path = [best_last]
    for bp_t in reversed(backpointers):
        path.append(bp_t[path[-1]])
    path.reverse()
    return path
        
from sklearn.model_selection import train_test_split

all_trace_ids = features['trace_id'].unique()
train_ids, test_ids = train_test_split(all_trace_ids, test_size=0.2, random_state=42)

df_train = features[features['trace_id'].isin(train_ids)]
df_test  = features[features['trace_id'].isin(test_ids)]

emissions   = fit_emission_model(df_train)
transitions = transition_matrix(df_train)

# test on one unseen trace
trace_id, trace = next(iter(df_test.groupby('trace_id')))
observations = trace.sort_values('step_id')['mean_current'].tolist()
path = viterbi(observations, emissions, transitions)

ground_truth = trace.sort_values('step_id')['cluster_kmer'].tolist()

print("Predicted: ", path)
print("Ground truth:", ground_truth)

# compare center cluster (position 2) of each kmer
pred_centers  = [k.split('_')[2] for k in path]
truth_centers = [k.split('_')[2] for k in ground_truth]

correct = sum(p == t for p, t in zip(pred_centers, truth_centers))
print(f"\nCenter-cluster accuracy: {correct}/{len(truth_centers)} = {correct/len(truth_centers):.2f}")


