# State-wise Neural Architecture Search
This repository provides the implementation of the method proposed in our paper [State-wise Neural Architecture Search](https://arxiv.org/pdf/2004.11178.pdf). Our method discovers competitive and high-performance architectures by exploring one order of magnitude fewer models compared to other approaches, as shown in the figure below. In addition, our method is the most resource-efficient as it designs architectures in a few hours on a single GPU.

<img src="/Figures/Main2.svg">

<img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="LICENSE" align="right">

## Requirements
- [Scikit-learn](http://scikit-learn.org/stable/)
- [Keras](https://github.com/fchollet/keras) (Recommended version 2.1.2)
- [Tensorflow](https://www.tensorflow.org/) (Recommended version 1.3.0 or 1.9)
- [Python 3](https://www.python.org/)

## Quick Start
[main.py](main.py) provides an example of our neural architecture search employing residual modules. We highlight that in this example, we are using only 2 epochs to discover convolutional architectures. However, in our paper, we use 200 epochs to discover the candidate architectures and each one is further trained for 100 epochs (see the experimental setup section in our paper for more details).

## Parameters
Our method takes two parameters:
1. Number of iterations (see line 314 in [main.py](main.py)).
2. Growth step (see line 312 in [main.py](main.py)).
## Additional parameters (not recommended)
1. Number of components of Partial Least Squares (see line 313 in [main.py](main.py))

## Results
The table below show the comparison between our architectures (using residual modules) with human-designed architectures. Please check our paper for more detailed results. We provide the models and weights to reproduce these results in [models/CIFAR10](models/CIFAR10).

| Architecture | Depth | Param. (Million) | FLOPs (Million) | Inference Time (Milliseconds) | Carbon Emission (kgCO_2eq) | Accuracy CIFAR-10 |
|:------------:|:-----:|:----------------:|:---------------:|:-----------------------------:|:--------------------------:|:-----------------:|
|   ResNet56   |   56  |       0.86       |       125       |             46.86             |            1.27            |       93.03       |
|  Ours (it=3) |   59  |       0.69       |       130       |             50.16             |            1.23            |       93.36       |
|   ResNet110  |  110  |        1.7       |       253       |             90.33             |            2.26            |       93.57       |
|  Ours (it=5) |   67  |       0.88       |       149       |             57.96             |            1.32            |       94.27       |

Please cite our paper in your publications if it helps your research.
```bash
@inproceedings{Jordao:2020,
author    = {Artur Jordao,
Fernando Yamada,
Maiko Lie and
William Robson Schwartz},
title     = {State-wise Neural Architecture Search},
booktitle = {International Conference on Pattern Recognition (ICPR). Accepted for publication.},
}
```
