import numpy as np
import tensorflow as tf



#dummy data
X = np.random.randn(1000, 20, 1).astype(np.float32)  # (samples, timesteps, features)
y = np.random.randint(0, 4, size=1000)    