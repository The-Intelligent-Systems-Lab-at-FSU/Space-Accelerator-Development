import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tifffile
import os

# Set the directory containing your images
image_dir = "bands_test/Aqua143/"

# Create an empty list to store the flattened image data
image_data = []

# Loop over each image file in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".tif"):
        # Load the image data and flatten it into a 2D array
        image = tifffile.imread(os.path.join(image_dir, filename))
        image_2d = image.reshape(-1, image.shape[-1])
        # Append the flattened image data to the list
        image_data.append(image_2d)

# Concatenate the flattened image data into a single 2D array
all_image_data = np.concatenate(image_data, axis=0)

# Apply the t-SNE algorithm to the 2D array
tsne = TSNE(n_components=2, random_state=0)
all_image_tsne = tsne.fit_transform(all_image_data)

# Plot the t-SNE results
plt.scatter(all_image_tsne[:, 0], all_image_tsne[:, 1])
plt.show()
