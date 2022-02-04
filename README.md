# UvA_FACT2022
Code repository of team 41 for UvA FACT course 2022. Team member: Paul, Veli, Jenn, Shuai.

The original code of DECAF paper is [here]( https://github.com/vanderschaarlab/DECAF)

## Prerequisites

Code is compatible with Python version 3.8.* due to explicit version requirements of some of the project dependencies.

We use [DVC](https://dvc.org/) for storing trained moodels.

## Installation

```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## Downloading pretrained models

If you want to run the notebooks with pretrained models you need to download them first by running:

```
dvc pull
```

## Get results

A quick overview of results can be seen in experimnet_1.py and experiment_2.py
