import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from skimage import io

image = io.imread("path/to/image.jpg")
#Used to load the image data into a NumPy array

image_2d = image.reshape(-1, 3)
#fit the image data into a 2-D array
#The reshape() function flattens the 3-D image array into a 2-D array with number of rows equal to pixels in the image (3 for RGB)

tsne = TSNE(n_components=2, random_state=0)
image_tsne = tsne.fit_transform(image_2d)
#Use the t-SNE algorithm on the 2-D array

plt.scatter(image_tsne[:, 0], image_tsne[:, 1])
plt.show()
#plot and display the t-SNE results.
