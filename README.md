# AlexNet Feature Extraction
Transfer learning involves taking a pre-trained neural network and adapting the neural network to a new, different data set.

Depending on both:

* the size of the new data set, and
* the similarity of the new data set to the original data set
the approach for using transfer learning will be different. There are four main cases:

* new data set is small, new data is similar to original training data
* new data set is small, new data is different from original training data
* new data set is large, new data is similar to original training data
* new data set is large, new data is different from original training data

## Traffic sign classifier
Let's use Transfer Learning with AlexNet to train a traffic sign classifier.
Download the [training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580a829f_train/train.p) and [AlexNet weights](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy).

[//]: # (Image References)

[image1]: ./aws_gpu_training.png "Training on a Amazon GPU instance"

### Files
* **imagenet_inference.py** to verify that the network classifies the images correctly.
```
python imagenet_inference.py
```

* **traffic_sign_inference.py** show how we resize the image for the traffic sign classifier and how well the classifier performs on the example construction and stop signs.

* **feature_extraction.py**: in order to successfully classify our traffic sign images, we need to remove the final, 1000-neuron classification layer and replace it with a new, 43-neuron classification layer.

This is called feature extraction, because we're basically extracting the image features inferred by the penultimate layer, and passing these features to a new classification layer.

_Notes_ 
That being said, the output classes you see should be present in signnames.csv.

* **train_feature_extraction.py** training the new classification layer. This was done on a Amazon GPU instance and you can find the already trained model in this repository.
![Training][image1]


## Credits
This lab utilizes:

An implementation of AlexNet created by [Michael Guerzhoy and Davi Frossard](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)
AlexNet weights provided by the [Berkeley Vision and Learning Center](http://bvlc.eecs.berkeley.edu/)
Training data from the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)
AlexNet was originally trained on the [ImageNet database](http://www.image-net.org/).


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/sbatururimi/Transfer-Learning-Alexnet-Feature-Extraction/blob/master/LICENSE)

