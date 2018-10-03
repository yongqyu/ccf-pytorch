# CCF
A pytorch GPU implementation of follows.

NCF : He et al. "Neural Collaborative Filtering" at WWW'17

CCF : NCF based Convolutional Collaborative Filtering

This code only covers the Movielens 1M Dataset https://grouplens.org/datasets/movielens/.

Preprocessing by ```Preprocess.ipynb``` is necessary.

## The requirements are as follows:
1.python 3.5

2.pytorch 0.4.0


## Example to run:
```
python train.py --mode NCF
python train.py --mode CCF
```
