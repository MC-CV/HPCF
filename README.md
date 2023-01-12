# Hypergraph Projection Enhanced Collaborative Filtering

## Environment
The codes of HPCF are implemented and tested under the following development environment:

TensorFlow:
* python=3.6.12
* tensorflow=1.15.0
* numpy=1.19.5
* scipy=1.7.3
* tensorflow-determinism=0.3.0
## Datasets
We utilized three datasets to evaluate HPCF: <i>Yelp, MovieLens, </i>and <i>Amazon</i>. Following the common settings of implicit feedback, if user $u_i$ has rated item $v_j$, then the element $(u_i, v_j)$ is set as 1, otherwise 0. We filtered out users and items with too few interactions. The datasets are divided into training set, validation set and testing set by 7:1:2.

## How to Run the Code
Please unzip the datasets first. Also you need to create the `History/` and the `Models/` directories. The command to train HPCF on the Yelp/MovieLens/Amazon dataset is as follows. The commands specify the hyperparameter settings that generate the reported results in the paper.

* Yelp
```
python labcode_efficient.py --data yelp --gpu 0
```
* MovieLens
```
python labcode_efficient.py --data ml10m --keepRate 1.0 --reg 1e-3 --gpu 0
```
* Amazon
```
python labcode_efficient.py --data amazon --reg 1e-2 --gpu 0
```

