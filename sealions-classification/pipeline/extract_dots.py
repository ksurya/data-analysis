"""Extract sea lion locations from training images.

This program peforms step 1 of the pipeline and a one-time operation.
The data generated is stored in a CSV file per image

The following operations are performed
    - Identify annotated dots
    - Evaluate the neighborhood
"""

import os
import glob
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from skimage import io, color, feature
from skimage import img_as_float


# Directories
DATA_DIR = "./data"
TRAINING_IMGS_DIR = os.path.join(DATA_DIR, "Train")
TRAINING_DOTTED_IMGS_DIR = os.path.join(DATA_DIR, "TrainDotted")
OUPUT_DIR = os.path.join(DATA_DIR, "TrainDots")
N_PROCESSES = 5


def find_dots(pth1, pth2):
    # pth1 := training image path
    # pth2 := corresponding "dotted" image path

    # Load images
    img1 = img_as_float(io.imread(pth1))
    img2 = img_as_float(io.imread(pth2))
    
    # extract dotted regions
    img = np.abs(img1 - img2)
    m1 = color.rgb2gray(img1)
    m2 = color.rgb2gray(img2)
    img[m1 < 0.07] = 0
    img[m2 < 0.07] = 0
    
    # find blobs
    img_gray = color.rgb2gray(img)
    blobs = feature.blob_log(img_gray, threshold=0.02, min_sigma=3, max_sigma=4)
    
    # classify blobs (these threshold neeed ubyte image)
    def get_sealion_type(im, x, y):
        r, g, b = im[x, y, :]
        sealion = "unknown"
        if r > 200 and b < 50 and g < 50: # red
            sealion = "adult_male"
        elif r > 200 and b > 200 and g < 50: # magenta
            sealion = "sub_adult_male"
        elif r < 100 and b < 100 and 150 < g < 200: # green
            sealion = "pup"
        elif r < 100 and  100 < b and g < 100: # blue
            sealion = "juvenile"
        elif r < 150 and b < 50 and g < 100: # brown
            sealion = "adult_female"
        return sealion
    
    # store blob locations in a dataframe..
    img_name = os.path.split(pth1)[1]
    df = pd.DataFrame(columns=("image_name", "width", "height", "sealion_type"))
    for idx, blb in enumerate(blobs):
        sealion = get_sealion_type(img_as_ubyte(img2), blb[0], blb[1])
        df.loc[idx] = (img_name, blb[0], blb[1], sealion)
    
    # calculate neighborhood
    def counter(r1):
        max_dist = 150 # search within this radius (in pixel units)
        neighbors = dict(
            unknown=0,
            adult_male=0,
            sub_adult_male=0,
            pup=0,
            juvenile=0,
            adult_female=0)
        for idx, r2 in df.iterrows():
            dist = (r1.width-r2.width) ** 2 + (r1.height-r2.height) ** 2
            if dist <= max_dist ** 2 and dist > 0:
                neighbors[r2.sealion_type] += 1
        return neighbors

    neighbors = df.apply(counter, axis=1)
    neighbors_cols = ("unknown","adult_male","sub_adult_male","pup","juvenile","adult_female")
    for each in neighbors_cols:
        df["n_" + each] = neighbors.apply(lambda x: x[each])
    df["n_total"] = neighbors.apply(lambda x: sum([x[i] for i in neighbors_cols]))
    
    # save as csv
    name = img_name.rsplit(".", 1)[0]
    df.to_csv(os.path.join(OUPUT_DIR, name + ".csv"), index_label="seq_id")


def main():
    training_imgs = sorted(glob.glob(os.path.join(TRAINING_IMGS_DIR, "*.jpg")))
    training_dotted_imgs = sorted(glob.glob(os.path.join(TRAINING_DOTTED_IMGS_DIR, "*.jpg")))
    Parallel(n_jobs=N_PROCESSES)(
        delayed(find_dots)(p1, p2) for p1, p2 in zip(training_imgs, training_dotted_imgs))


if __name__ == "__main__":
    main()
