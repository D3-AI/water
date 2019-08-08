<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DaiLab” /><a href="https://www.xylem.com/" target="_blank"><img width=15% src="https://upload.wikimedia.org/wikipedia/en/thumb/2/2d/Xylem_Logo.svg/1920px-Xylem_Logo.svg.png" alt="XylemLogo"/></a>
<br/>
<i>A collaborative open source project between Data to AI Lab at MIT and Xylem Inc..</i>
</p>

[![PyPI Shield](https://img.shields.io/pypi/v/water.svg)](https://pypi.python.org/pypi/ml-water)
[![Travis CI Shield](https://travis-ci.org/D3-AI/water.svg?branch=master)](https://travis-ci.org/D3-AI/water)

# Water

Machine learning for internet of things related to water systems.

- Free software: MIT license
- Documentation: https://D3-AI.github.io/water
- Homepage: https://github.com/D3-AI/water

# Overview

The Water project is a collection of end-to-end solutions for machine learning tasks commonly found in monitoring water distribution and delivery networks. Most tasks utilize sensor data emanating from monitoring systems. We utilize the foundational innovations developed for automation of machine Learning at Data to AI Lab at MIT. This project is developed in close collaboration with Xylem Inc. 

The salient aspects of this customized project are:
* A set of ready to use, well tested pipelines for different machine learning tasks. These are vetted through testing across multiple publicly available datasets for the same task. 
* An easy interface to specify the task, pipeline, and generate results and summarize them. 
* A production ready, deployable pipeline. 
* An easy interface to ``tune`` pipelines using Bayesian Tuning and Bandits library. 
* A community oriented infrastructure to incorporate new pipelines. 
* A robust continuous integration and testing infrastructure. 
* A ``learning database`` recording all past outcomes --> tasks, pipelines, outcomes. 

## Concepts

Before diving into the software usage, we briefly explain some concepts and terminology.

### Primitive

We call the smallest computational blocks used in a Machine Learning process
**primitives**, which:

* Can be either classes or functions.
* Have some initialization arguments, which MLBlocks calls `init_params`.
* Have some tunable hyperparameters, which have types and a list or range of valid values.

### Template

Primitives can be combined to form what we call **Templates**, which:

* Have a list of primitives.
* Have some initialization arguments, which correspond to the initialization arguments
  of their primitives.
* Have some tunable hyperparameters, which correspond to the tunable hyperparameters
  of their primitives.

### Pipeline

Templates can be used to build **Pipelines** by taking and fixing a set of valid
hyperparameters for a Template. Hence, Pipelines:

* Have a list of primitives, which corresponds to the list of primitives of their template.
* Have some initialization arguments, which correspond to the initialization arguments
  of their template.
* Have some hyperparameter values, which fall within the ranges of valid tunable
  hyperparameters of their template.

A pipeline can be fitted and evaluated using the MLPipeline API in MLBlocks.


## Current tasks and pipelines

In our current phase, we are addressing two tasks - time series classification and time series regression. To provide solutions for these two tasks we have two components. 

### TimeSeriesEstimator

This class is the one in charge of learning from the data and making predictions by building
[MLBlocks](https://hdi-project.github.io/MLBlocks) and later on tuning them using
[BTB](https://hdi-project.github.io/BTB/)

This class comes in two flavours in the form of subclasses, the **TimeSeriesClassifier** and the
**TimeSeriesRegressor**, to be used in the corresponding problem types.

### TimeSeriesLoader

A class responsible for loading the time series data from CSV files, and return it in the
format ready to be used by the **TimeSeriesEstimator**.




### Time series dataset

A dataset is a folder that contains time series data and information about
a Machine Learning problem in the form of CSV and JSON files.

The expected contents of the `dataset` folder are:

* A `metadata.json` with information about all the tables found in the dataset. This file follows the [Metadata.json schema](https://github.com/HDI-Project/MetaData.json) with three small modifications:
  * The root document has a `name` entry, with the name of the dataset.
  * The foreign key columns are be of type `id` and subtype `foreign`.
  * The `datetime` columns that are time indexes need to have the `time_index` subtype.

* A CSV file containing the training samples with, at least, the following columns:
  * A unique index
  * A foreign key to at least one timeseries table
  * A time index that works as the cutoff time for the training example
  * If the problem is supervised, a target column.

Then, for each type of timeseries that exist in the dataset, there will be:

* A CSV file containing the id of each timeseries and any additional information associated with it
* A CSV file containing the timeseries data with the following columns:
  * A unique index
  * A foreign key to the timeseries table
  * A time index
  * At least a value column

### Tuning

We call tuning the process of, given a dataset and a template, find the pipeline derived from the
given template that gets the best possible score on the given dataset.

This process usually involves fitting and evaluating multiple pipelines with different hyperparameter
values on the same data while using optimization algorithms to deduce which hyperparameters are more
likely to get the best results in the next iterations.

We call each one of these tries a **tuning iteration**.


# Getting Started

## Installation

The simplest and recommended way to install **Water** is using pip:

```bash
pip install ml-water
```

For development, you can also clone the repository and install it from sources

```bash
git clone git@github.com:D3-AI/water.git
cd water
make install-develop
```

## Usage Example

In this example we will load some demo data using the **TimeSeriesLoader** and fetch it to the
**TimeSeriesClassifier** for it to find the best possible pipeline, fit it using the given data
and then make predictions from it.

### Load and explore the data

We first create a loader instance, passing the path to the dataset, the name of the column
that we want to predict, and the name of the table where this column can be taken from.


```python
from water.loader import TimeSeriesLoader

loader = TimeSeriesLoader(
    dataset_path='examples/datasets/ItalyPowerDemand',
    target_table='demand',
    target_column='target'
)
```

Then we call the `loader.load` method, which will return three elements:

* `X`: The contents of the target table, where the training examples can be found, without the target column.
* `y`: The target column, as extracted from the target table.
* `data`: A dictionary containing the additional elements that the Pipeline will need to run, including the actual time series data.


```python
X, y, data = loader.load()
X.head(5)
```

<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>demand_id</th>
      <th>timeseries_id</th>
      <th>cutoff_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2010-01-25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2010-01-25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>2010-01-25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>2010-01-25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>2010-01-25</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.head(5)
```




    0    1
    1    1
    2    2
    3    2
    4    1
    Name: target, dtype: int64




```python
data.keys()
```




    dict_keys(['entities', 'relationships', 'target_entity', 'target_column', 'dataset_name'])



### Split the data

If we want to split the data in train and test subsets, we can do so by splitting the `X` and `y` variables.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

### Finding the best Pipeline

Once we have loaded the data, we create a **TimeSeriesClassifier** instance and call its `tune` method to
find the best possible pipeline for our data.

We start by importing the `TimeSeriesClassifier` and creating an instance.


```python
from water.estimators import TimeSeriesClassifier

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

tsc = TimeSeriesClassifier()
```

We then pass the `X` and `y` partitions and the `data` dictionary, along with the number of tuning iterations
that we want to perform to the `tune` method, which will run the indicated number of iterations
trying to figure out the best possible hyperparameters.


```python
tsc.tune(X_train, y_train, data, iterations=5)
```

After the tuning process has finished, the hyperparameters have been already set in the classifier.

We can see the found hyperparameters by calling the `get_hyperparameters` method


```python
tsc.get_hyperparameters()
```




    {'mlprimitives.preprocessing.ClassEncoder#1': {},
     'featuretools.dfs#1': {'encode': True,
      'max_depth': 2,
      'remove_low_information': False},
     'sklearn.preprocessing.Imputer#1': {'missing_values': 'NaN',
      'axis': 0,
      'copy': True,
      'strategy': 'mean'},
     'xgboost.XGBClassifier#1': {'n_jobs': -1,
      'n_estimators': 409,
      'max_depth': 8,
      'learning_rate': 0.45727735286952875,
      'gamma': 0.7661016859076536,
      'min_child_weight': 2},
     'mlprimitives.preprocessing.ClassDecoder#1': {}}



as well as the obtained cross validation score by looking at the `score` attribute of the `tsc` object


```python
tsc.score
```




    0.6212121212121213



Once we are satisfied with the obtained cross validation score, we can proceed to call
the `fit` method passing again the same data elements.


```python
tsc.fit(X_train, y_train, data)
```

After this, we are ready to make predictions on new data


```python
predictions = tsc.predict(X_test, data)
predictions[0:5]
```




    array([2, 1, 2, 2, 2])



### Trying new templates

The **TimeSeriesClassifier** and **TimeSeriesRegressor** have a default template for each problem.

This template can be overriden by passing a new template dictionary when the instance is created.


```python
template = {
    'primitives': [
        'featuretools.dfs',
        'sklearn.preprocessing.Imputer',
        'sklearn.preprocessing.StandardScaler',
        'sklearn.ensemble.RandomForestClassifier',
    ],
    'init_params': {
        'featuretools.dfs#1': {
            'encode': True
        }
    }
}

tsc = TimeSeriesClassifier(template=template)
tsc.tune(X_train, y_train, data, iterations=5)
tsc.score
```




    0.5404040404040403




```python
tsc.fit(X_train, y_train, data)
predictions = tsc.predict(X_test, data)
predictions[0:5]
```




    array([1, 1, 1, 1, 1])


## What's next?
￼ ￼
￼For more details about **water** and all its possibilities and features, please check the
￼[project documentation site](https://D3-AI.github.io/water/)!
