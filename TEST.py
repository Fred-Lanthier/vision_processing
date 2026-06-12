import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt
import time

# Generate two example time series of different lengths
reference = np.sin(np.linspace(0, 6.28, num=100))
query = np.sin(np.linspace(0, 6.28, num=90)) + 0.1

# Compute the alignment
start = time.perf_counter()
alignment = dtw(query, reference, keep_internals=True)
end = time.perf_counter()
print(f"DTW computation time: {end - start:.4f} seconds")
# Extract key metrics
print(f"DTW Distance: {alignment.distance}")
print(f"Optimal Alignment Path (Query indices): {alignment.index1}")
print(f"Optimal Alignment Path (Reference indices): {alignment.index2}")

# Plot the alignment (requires matplotlib)
alignment.plot(type="twoway")
plt.show()