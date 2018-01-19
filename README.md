# CNNForSentenceClassification

## Overview
This is a small project which implemented based on the paper [Yoon Kim, "Convolutional Neural Networks for Sentence Classification", 2014.](https://arxiv.org/abs/1408.5882).

## Requirements
* Python 2.7.12
* [Keras](https://keras.io/): Using Tensorflow backend
* [word2vec](https://github.com/danielfrg/word2vec): Python wrapper for [Tensorflow's word2vec algorithm](https://www.tensorflow.org/tutorials/word2vec)  
*For more requirement packages, see the file [requirement.txt](requirement.txt) or using python pip*
```
pip install -r requirements.txt
```

## Dataset Information
The data was first used in Bo Pang and Lillian Lee,
["Seeing stars: Exploiting class relationships for sentiment categorization
with respect to rating scales.", Proceedings of the ACL, 2005.](http://www.cs.cornell.edu/people/pabo/movie-review-data). See the file [rt-polaritydata.README.1.0.txt](data/rt-polaritydata.README.1.0.txt) for more information.

## License
This project is licensed under the MIT License.