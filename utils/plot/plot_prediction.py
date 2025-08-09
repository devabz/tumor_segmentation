import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.utils import dice_score

def plot_prediction(mip: np.ndarray, seg: np.ndarray, seg_pred: np.ndarray):

    score = dice_score(seg,seg_pred)
    plt.figure(figsize=(9.2,3))

    plt.subplot(1,4,1)
    plt.imshow(mip)
    plt.axis("off")
    plt.title("PET MIP")

    plt.subplot(1,4,2)
    plt.imshow(seg)
    plt.axis("off")
    plt.title("True Segmentation")

    plt.subplot(1,4,3)
    plt.imshow(seg_pred)
    plt.axis("off")
    plt.title("Predicted Segmentation")

    TP = ((seg_pred>0)&(seg>0))[:,:,:1]
    FP = ((seg_pred>0)&(seg==0))[:,:,:1]
    FN = ((seg_pred==0)&(seg>0))[:,:,:1]
    img = np.concatenate((FP,TP,FN),axis=2).astype(np.uint8)*255

    plt.subplot(1,4,4)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"dice score = {score:.02f}")
    plt.legend(["a","b"])

    # Create green, red, and blue squares as proxy artists
    green_square = mpatches.Patch(color='green', label='TP')
    red_square = mpatches.Patch(color='red', label='FP')
    blue_square = mpatches.Patch(color='blue', label='FN')

    # Add the proxy artists to the legend
    plt.legend(handles=[green_square, red_square, blue_square],loc="lower right")
    plt.tight_layout(h_pad=2,w_pad=0,pad=1.5)
    plt.show()
