# Introduction

This repository cotains a new dataset to locate and identify leaks in Water Distribution Systems through realistic simulations performed in EPANET-matlab-toolkit. 

GraphLeak consists of simulations in different scenarios and topologies designed under realistic hydraulic parameters where each node represents a measurement
point. 

### Evaluation

The results obtained by a Multi-layer Perceptron are evaluated by the ain classification metrics of confusion matrix, such as accuracy, precision, reacall and F1-score.

The Mean Absolute Error (MAPE) is used to analyze the error between predictions and the correct values.

# PreProcess python file


### Prerequisites
- Python3
- [PyTorch](http://pytorch.org)
- All the libraries in <code>requirements.txt</code>

# Raw Data Download

All the contents of GraphLeak are public and can be acessed [here](https://googledrive.com/)

### Data generation

From raw Data, generate the dataset by running:

<pre><code> python3 main.py </pre></code>

### configurations

Noise - If you want standard noise in the data, please set noise as True
- <code>attention: True</code>

- Noise specification
- Nodes Normalization
- Data Normalization


