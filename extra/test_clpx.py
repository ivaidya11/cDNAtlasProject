import numpy as np
import random
import matplotlib.pyplot as plt
from clpx import predict_squiggle

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

random.seed(42)
seq = ''.join(random.choices(AMINO_ACIDS, k=50))

print(f"Random peptide ({len(seq)} aa):")
print(seq)

I_open_pA = 220  # typical open pore current in pA — adjust to your experiment

squiggle = predict_squiggle(seq) * I_open_pA

print(f"\nSquiggle length: {len(squiggle)}")
print(f"Min: {squiggle.min():.2f} pA  Max: {squiggle.max():.2f} pA  Mean: {squiggle.mean():.2f} pA")
print(f"Peak-to-peak variation: {squiggle.max() - squiggle.min():.2f} pA")

plt.figure(figsize=(10, 4))
plt.plot(squiggle)
plt.title(f"Predicted nanopore current\n{seq}")
plt.xlabel("Position")
plt.ylabel("Predicted current (pA)")
plt.tight_layout()
plt.savefig("test_clpx_squiggle.png", dpi=150)
plt.show()
print("\nPlot saved to test_clpx_squiggle.png")
