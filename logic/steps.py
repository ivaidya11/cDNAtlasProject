import poreflow as pf
from poreflow.steps import changepoint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from logic.database import AA_CLASS

df = pd.read_csv('../features.csv')

data_path = "../nanopore_trace_database.csv"

db = pd.read_csv(data_path)
SAMPLING_FREQ = 5000 # needs to match what i used to generate


trace_id = 0
trace = db[db['trace_id'] == trace_id]
current = trace['current_pA'].to_numpy()
time_ms = trace['time_ms'].to_numpy()

steps = changepoint.get_steps(current, sfreq=SAMPLING_FREQ, sensitivity=0.3, min_level_length=10)
