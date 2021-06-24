# Self-supervised depth estimation on synthesized stereo image-pairs using a cycle-consistent generative adversarial network

*This file contains general notes and information that are important for understanding and using this program.*

This deep learning project is for stereo image-pair synthesis and depth estimation on a given stereo image-pair. In this project, stereo image-pair synthesis is called left-to-right, i.e **L2R** and depth estimation is called stereo-to-depth, i.e. **S2D**. Hence, there are two sub-folders in the dataset folder and these contain one example dataset for L2R synthesis and S2D estimation. Within each sub-folder there's a train and a test folder. For each train or test dataset there are two domains, A and B. Both L2R and S2D networks learn to map images from domain A to domain B, and vice versa. For L2R synthesis it is recommended to use paired data, but not required. For S2D estimation it is not required, because the network is able to deal with un-paired data quite well.

The outputs, results and weights folder should also have a "l2r" and "s2d" directory. Usually the code should make those directories automatically, if they are not present: do it manually.  These folders are empty, but will be filled with data once you run the code.

## Environment

* Anaconda virtual environment 
* Python 3.7 

## Python packages

*Assuming a fresh Anaconda environment.*

* PyTorch
* PyTorch_fid
* Torchvision
* Matplotlib
* Numpy
* Scipy
* Yaml
* Tqdm
* Labml-helpers