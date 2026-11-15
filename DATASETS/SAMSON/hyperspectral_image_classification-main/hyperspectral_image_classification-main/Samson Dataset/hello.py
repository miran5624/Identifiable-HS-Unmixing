import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Load the MATLAB file
mat_data = scipy.io.loadmat('samgt.mat')
A = mat_data['A']
M = mat_data['M']

# --- 1. Plot the Endmember Spectral Signatures ---
plt.figure(figsize=(10, 4))
plt.plot(M[:, 0], label='Material 1 (e.g., Soil)')
plt.plot(M[:, 1], label='Material 2 (e.g., Tree)')
plt.plot(M[:, 2], label='Material 3 (e.g., Water)')
plt.title('Spectral Signatures of the 3 Materials')
plt.xlabel('Spectral Band (1 to 156)')
plt.ylabel('Reflectance')
plt.legend()
plt.grid(True)
plt.show()

# --- 2. Plot the Abundance Maps (The Images) ---
# Reshape the 9025 pixels back into a 95x95 grid
spatial_size = 95 

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['Material 1 Abundance', 'Material 2 Abundance', 'Material 3 Abundance']

for i in range(3):
    # Take the i-th row and reshape it to 95x95
    abundance_map = A[i, :].reshape((spatial_size, spatial_size))
    
    # Show the image map
    im = axes[i].imshow(abundance_map, cmap='jet')
    axes[i].set_title(titles[i])
    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()