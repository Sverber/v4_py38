# Self-supervised single image depth estimation using a cycle consistent generative adversarial network

*This file contains general notes and information that deem important for understanding and using this program.*

## Environment

* Anaconda virtual environment 
* Python 3.8.5 64-bit

## Python packages

* ***Package list coming soon..***
* *Add package..*
* *Add package..*

## Datasets

* ### **horse2zebra_000_999**       
    * Contains **1000** training images per class.
    * Contains **120** test images per class.
    * Original image dimensions are **256 pixels** in width by **256 pixels** in height using **3 channels**.
    * Resize to 61%: (156, 156) pixels.

* ### **kitti_synthesized_000_999** 
    * Contains **1000** training images per class.
    * Contains **0** test images per class.
    * Original image dimensions are **1216 pixels** in width by **352 pixels** in height using **3 channels**.
    * Resize to 10%: (122, 35) pixels.

* ### **DrivingStereo_demo_images** 
    * Contains **300** training pairs of stereo images with the corresponding disparity map.
    * Contains **0** test pairs of stereo images with the corresponding disparity map.
    * Original image dimensions are **1762 pixels** in width by **800 pixels** in height using **3 channels**.
    * Resize to 10%: (176, 79) pixels.

* ### **Test_Set**
    * Contains **600** training pairs of stereo images and their corresponding disparity map.
    * Contains **76** test pairs of stereo images and their corresponding disparity map.
    * Original image dimensions are **2208 pixels** in width by **1242 pixels** in height using **3 channels**.
    * Resize to xx%: (68, 120) pixels.

* ### **UASOL**
    * ***Under construction***
    * Contains **0** training pairs of stereo images and their corresponding disparity map.
    * Contains **0** test pairs of stereo images and their corresponding disparity map.
    * Original image dimensions are **2208 pixels** in width by **1242 pixels** in height using **3 channels**.
    * Resize to 5.43%: (68, 120) pixels. *(divide by 15 and multiply with 0.82)*