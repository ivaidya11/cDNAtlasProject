"""

Parallel version of database.py — generates traces using all available CPU cores.
Imports all simulation logic from database.py; only generate_database is replaced.

Usage:
    python logic/database_parallel.py

"""

from joblib import Parallel, delayed
import pandas as pd
import random

from database import (
    simulate_trace,
    plot_example_trace,
    AA_LIST,
    N_TRACES,
    PEP_LENGTH,
    IOS,
    ALPHA,
    BETA,
    SIGMA_WHITE,
    SIGMA_FLICKER,
)


def _simulate_one(trace_id: int, pep_length: int) -> pd.DataFrame:
    peptide = ''.join(random.choices(AA_LIST, k=pep_length))
    trace_df = simulate_trace(peptide)
    trace_df['trace_id'] = trace_id
    trace_df['peptide_sequence'] = peptide
    return trace_df


def generate_database(n_traces: int, pep_length: int, output_path: str, n_jobs: int = -1) -> pd.DataFrame:
    """
    Generate a database of simulated traces in parallel, writing to CSV in chunks
    to avoid a slow pd.concat over 30k DataFrames at the end.
    n_jobs=-1 uses all available CPU cores.
    """
    CHUNK = 1000
    print(f"Generating {n_traces} traces in parallel (n_jobs={n_jobs}), writing every {CHUNK}...")

    first_write = True
    for start in range(0, n_traces, CHUNK):
        ids = range(start, min(start + CHUNK, n_traces))
        chunk_traces = Parallel(n_jobs=n_jobs)(
            delayed(_simulate_one)(trace_id, pep_length) for trace_id in ids
        )
        chunk_df = pd.concat(chunk_traces, ignore_index=True)
        chunk_df.to_csv(output_path, mode='w' if first_write else 'a',
                        header=first_write, index=False)
        first_write = False
        print(f"  Written {min(start + CHUNK, n_traces)} / {n_traces} traces")

    print(f"\nDatabase saved to: {output_path}")
    database = pd.read_csv(output_path)
    print(f"Total rows:  {len(database)}")
    print(f"Columns:     {list(database.columns)}")
    return database


if __name__ == "__main__":
    output_path = "data/nanopore_trace_database_BIG.csv"

    print(f"Generating {N_TRACES} traces (peptide length = {PEP_LENGTH})...")
    print(f"IOS = {IOS} pA | alpha = {ALPHA} | beta = {BETA} | sigma_white = {SIGMA_WHITE} | sigma_flicker = {SIGMA_FLICKER}\n")

    db = generate_database(N_TRACES, PEP_LENGTH, output_path)
    plot_example_trace(db, trace_id=500)

    print("\n--- Current (pA) by region ---")
    print(db.groupby('region')['current_pA'].describe().round(2))
