import cv2
import matplotlib.pyplot as plt

def plot_grid(images, titles=None, ncols=3, nrows=3, figsize=(15, 10), resize: tuple = None):
    """
    Plots a grid of images with optional titles.

    Parameters:
    - images: List of images to plot.
    - titles: List of titles for each image (optional).
    - ncols: Number of columns in the grid.
    - nrows: Number of rows in the grid.
    - figsize: Size of the figure.
    """
    
    if isinstance(resize, tuple) and len(resize) == 2:
        images = [cv2.resize(img, resize) for img in images]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            if titles and i < len(titles):
                ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()