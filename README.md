# Odyssey

A modular, customizable Bayesian Optimization software module for single and multi-objective optimization.

See also the [original repository](https://gitlab.com/auto_lab/odyssey) and its [documentation website](https://odyssey-51edb0.gitlab.io/).

## Getting Started

### Dependencies

* python>=3.9.21
* torch>=2.2.2
* numpy==1.26.4
* pandas==2.2.2
* botorch==0.10.0

NOTE: for the SMAC and Gryffin interfaces python==3.9.21 is required 

### Installing

We strongly suggest installing Odyssey in a conda or python virtual environment. For example

```
conda create -n odyssey python=3.10
conda activate odyssey
```

Clone the repo and change into the base directory. To install the base version of 
Odyssey with BOTorch navigators use

```
pip install .
```

### Usage

Fundamentally an optimisation with Odyssey involves:
* an `Objective`, which describes the function that will be optimized and processes the inputs and outputs
* a `Mission`, which details the parameter space and optimisation settings and maintains the collected data
* a `Navigator`, which suggests the next point to sample in the optimisation

See the examples in the scripts directory.
