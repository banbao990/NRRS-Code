total_pixels = 643460
filename = R"images/bathroom-d6-{}.bin".format(total_pixels)

import numpy as np
import matplotlib.pyplot as plt

with open(filename, "rb") as f:
    data = f.read()

data_array = np.frombuffer(data, dtype=np.float32)
print(data_array.shape, total_pixels)

# min, max, mean, std
print(np.min(data_array), np.max(data_array), np.mean(data_array), np.std(data_array))

percentiles = [5, 20, 50, 80, 95]
for p in percentiles:
    print(f"{p}th percentile: {np.percentile(data_array, p)}")

# histogram
plt.hist(data_array, bins=1000)
plt.yscale("log")
plt.show()
