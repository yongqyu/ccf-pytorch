# Convolutional Collaborative Filtering (CCF)
A pytorch GPU implementation of follows.

NCF : He et al. "Neural Collaborative Filtering" at WWW'17.

ONCF : He et al. "Outer Product-based Neural Collaborative Filtering" at IJCAI'18.

CCF : CNN based NCF.

This code only covers the Movielens 1M Dataset https://grouplens.org/datasets/movielens/.

Preprocessing by ```Preprocess.ipynb``` is necessary.

## The requirements are as follows:
1.python 3.5

2.pytorch 0.4.0


## Example to run:
```
python train.py --mode NCF
python train.py --mode ONCF
python train.py --mode CCF
```
