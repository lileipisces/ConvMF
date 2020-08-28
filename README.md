# ConvMF
re-implementation of [ConvMF (RecSys'16)](https://github.com/cartopy/ConvMF) in TensorFlow with even better performance

## Introduction
Our implementatation achieves 0.848 in terms of RMSE (check the log.txt) on the [ML-1M dataset](http://dm.postech.ac.kr/~cartopy/ConvMF/) provided by the authors while their reported result is only 0.8549.

Each line in the input text is formatted as:

**user::item::rating::word1 word2 word3**

Example:

**BCB7302F3A2AD466E27937439CF8AF8C::305921::5::ask for a view of the harbour ! 14th floor was very quiet . come to the kowloon side .**

## Paper:

> Kim, Donghyun, et al. Convolutional matrix factorization for document context-aware recommendation. RecSys'16.
