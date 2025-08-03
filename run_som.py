
import matplotlib.pyplot as plt
import numpy as np
import som as sm

# -------------------------
# Example usage / testing
# -------------------------

np.random.seed(42)

# Generate 100 3D input vectors (already scaled between 0–1)
data = np.random.rand(100, 3)

# Create SOM with 10x10 grid and train for 100 iterations
som = sm.SelfOrganizingMap(width=10, height=10, input_dim=3, alpha=0.1, n_iterations=100)
som.train(data)

# Retrieve trained weights for visualization
weights = som.get_weights()

# Save the trained SOM grid as an image (interpreting weights as RGB)
plt.figure(figsize=(6, 6))
plt.imshow(weights)
plt.axis('off')
plt.savefig('som_output.png', bbox_inches='tight', pad_inches=0)
