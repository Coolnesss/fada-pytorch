# FADA-pytorch

PyTorch implementation of Few-Shot Domain Adaptation (https://arxiv.org/abs/1711.02536)

Disclaimer: I'm not in any way affiliated with the authors. There might be errors in the implementation.

### Todo

* [x] Main algorithm implemented
* [ ] Fix TODOs in code
* [ ] Comprehensive tests 
* [ ] More datasets to test on (currently only MNIST -> USPS)
* [ ] The authors don't give the value of γ they used, one should be found via cross-validation

### Usage

Use `python3 main.py` to run the MNIST -> SVHN training and print accuracy at each epoch.

### Tests

Preliminary results show ~46% accuracy on test set with `n=7` samples per class from target domain. This approximately matches what is reported in the paper (47.0%)