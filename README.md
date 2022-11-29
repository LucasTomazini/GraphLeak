# Introduction

This repository cotains a new dataset to locate and identify leaks in Water Distribution Systems through realistic simulations performed in EPANET-matlab-toolkit. 

GraphLeak consists of simulations in different scenarios and topologies designed under realistic hydraulic parameters where each node represents a measurement
point. 

### Evaluation

The results obtained by a Multi-layer Perceptron are evaluated by the ain classification metrics of confusion matrix, such as accuracy, precision, reacall and F1-score.

The Mean Absolute Error (MAPE) is used to analyze the error between predictions and the correct values.

# Raw Data Download

All the contents of GraphLeak are public and can be acessed [here](https://googledrive.com/)

# PreProcess python file

### Prerequisites
- Python3
- [PyTorch](http://pytorch.org)
- All the libraries in <code>requirements.txt</code>



### Data generation

From raw Data, generate the dataset by running:

<pre><code> python3 main.py </pre></code>

### Configurations

**Noise** - If you want a Gaussian noise in the data, set noise as True.
- <code>Noise: True</code>

**Noise specification** - If there is noise in the data, specify the configuration bellow:
- <code>mu: 0 </code> mean default
- <code>sigma: 0.1 </code> standard deviation default

**Nodes Normalization** - Set True (recommended) to normalize values between nodes.
- <code>Node_normalization: True</code>

**Data Normalization** - Set True (recommended) to normalize values in the range 0 to 1.
- <code>Data_normalization: True</code>

