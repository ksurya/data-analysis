Counting Sea Lions by their kind in an aerial image
===================================================

Kaggle competition: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count

Update on Apr 04, 2018
======================
The below notes is not latest and not fully reflect the work

Pipeline:
=========

Step 1: Capture sea lions locations in training images
------------------------------------------------------
    -> Take difference of normal and dotted images
    -> Apply threshold to discard non-dot colors
    -> Use blob detection to identify dot (sea lion) locations
    -> Find neighbors of all sea lions grouped by their type
    
    -> Performance: Use Spark (eg: Google Dataproc)
    -> This is one time operation
    
    -> Evaluation: Validate the correctness with input count data

Step 2: Segmentation of sea lions in training images
----------------------------------------------------
    Approach (a) proposed by Siddharth
    -> Take a 30x30 region around a dot
    -> Create a template for each orientation of sea lion
    -> Determine the likelihood of sealion in a region using KNN
    
    Approach (b)
    -> Take a 120x120 region around a dot
    -> Use hessian filter to identify continuous edges in the region
    -> Apply marching squares to identify contours
    -> Determine the contour in which sea lion exists [needs improvement]
        -> Dot location should be inside the contour region
        -> Contour's area is not more / less than "x" percent
        -> Look at the curvature of the contour ...
    -> For pup sea lions, simply take a dxd region around the dot location
    -> [experiment] Use HSV color space, remove intensite etc.. 

    -> Performance: Use Spark (eg: Google Dataproc) to cache all points in contours
        in SQLite or CSV file

Step 3: Detection of sea lions [testing / validation]
-----------------------------------------------------
    One could simply use KNN (as mentioned above) for the sake of 
    building a basic pipeline. Followed by building a CNN model. 
    Need to make progress here...

