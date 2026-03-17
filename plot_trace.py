import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def plot_trace(database: pd.DataFrame, trace_id: int = 0, IOS: float = 240, ax=None):
    """
    Plot a single trace from the database, coloured by region.
    Pass an existing ax to overlay onto it (e.g. for adding step levels).
    """
    trace = database[database['trace_id'] == trace_id]
    peptide = trace['peptide_sequence'].iloc[0]

    colors = {
        'DNA':        'steelblue',
        'linker':     'green',
        'peptide':    'orange'
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
        standalone = True
    else:
        standalone = False

    for region, group in trace.groupby('region', sort=False):
        ax.plot(group['time_ms'], group['current_pA'],
                label=region, color=colors[region], linewidth=0.8)

    ax.axhline(IOS, color='gray', linestyle='--', alpha=0.4, label='IOS reference')
    ax.set_title(f'Trace {trace_id} — Peptide: {peptide}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (pA)')
    ax.legend()
    ax.grid(True)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax