# ConvMF
re-implementation of [ConvMF+ (RecSys'16)](https://github.com/cartopy/ConvMF) in TensorFlow with even better performance

Our implementatation achieves 0.848 in terms of RMSE (check the log.txt) on the ML-1M dataset provided by the authors while their reported result is only 0.8549.

Each line in the input text is formatted as:

**user::item::rating::word1 word2 word3**

Example:

**BCB7302F3A2AD466E27937439CF8AF8C::305921::5::ask for a view of the harbour ! 14th floor was very quiet . come to the kowloon side .**

Original paper:

Kim, Donghyun, et al. "Convolutional matrix factorization for document context-aware recommendation." Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 2016.


We appreciate it if you cite our paper when using the code:

@inproceedings{RSBD19-CCANN,
	title = {Context-aware Co-Attention Neural Network for Service Recommendations},
	author = {Li, Lei and Dong, Ruihai and Chen, Li},
	booktitle = {Proceedings of ICDE'19 Workshop on Recommender Systems with Big Data},
	year = {2019},
	organization = {IEEE}
}
