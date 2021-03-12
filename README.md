# Self-supervised single image depth estimation using a cycle consistent generative adversarial network

*This file contains general notes and information that deem important for understanding and using this program.*

## Environment

* Anaconda environment 
* Python 3.8.5 64-bit

## Package

* Add packages..

## Datasets

* ### **horse2zebra_000_999**       
    * Contains 1000 training images per class.
    * Contains 120 test images per class.
    * Original image size is 256 x 256 pixels.
    * Resize to 61%: (156, 156) pixels.
    * Rncrop to 82%: (128, 128) pixels.

* ### **kitti_synthesized_000_999** 
    * Contains 1000 training images per class.
    * Contains 0 test images per class.
    * Original image size is 1216 x 352 pixels.
    * Resize to 10%: (122, 35) pixels.
    * Rncrop to 82%: (100, 28) pixels.

"""