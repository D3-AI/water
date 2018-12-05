# -*- coding: utf-8 -*-
import logging
import warnings
from collections import defaultdict

import numpy as np
from btb import HyperParameter
from btb.tuning import GP
from mlblocks import MLPipeline
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings('ignore', category=DeprecationWarning)

LOGGER = logging.getLogger(__name__)


class TimeSeriesEstimator(object):
    template = {}
    maximize_score = None
    cv_class = None

    def __init__(self, scorer, cv=None, cv_splits=None, random_state=0,
                 template=None, hyperparameters=None, *args, **kwargs):

        if not (cv or cv_splits):
            raise ValueError('Either a CV object or a cv_split must be provided.')

        if cv and not callable(getattr(cv, 'split', None)):
            raise ValueError('The cv object must have an `split` method')

        self.cv = cv or self.cv_class(n_splits=cv_splits, shuffle=True, random_state=random_state)
        self.random_state = random_state
        self.scorer = scorer
        self.fitted = False
        self.tuner = None

        template = template or self.template
        self.pipeline = MLPipeline.from_dict(template)

        if hyperparameters:
            self.set_hyperparameters(hyperparameters)

    def get_hyperparameters(self):
        return self.pipeline.get_hyperparameters()

    def set_hyperparameters(self, hyperparameters):
        self.pipeline.set_hyperparameters(hyperparameters)
        self.fitted = False

    def get_tunable_hyperparameters(self):
        return self.pipeline.get_tunable_hyperparameters()

    @staticmethod
    def clone_pipeline(pipeline):
        return MLPipeline.from_dict(pipeline.to_dict())

    def _is_better(self, score, best_score):
        if self.maximize_score:
            return score > best_score

        return score < best_score

    def get_tunables(self):
        tunables = []
        tunable_keys = []
        for block_name, params in self.pipeline.get_tunable_hyperparameters().items():
            for param_name, param_details in params.items():
                key = (block_name, param_name)
                param_type = param_details['type']
                param_type = 'string' if param_type == 'str' else param_type

                if param_type == 'bool':
                    param_range = [True, False]
                else:
                    param_range = param_details.get('range') or param_details.get('values')

                value = HyperParameter(param_type, param_range)
                tunables.append((key, value))
                tunable_keys.append(key)

        return tunables, tunable_keys

    def score_pipeline(self, pipeline, X, y, data, time_index, index):
        scores = []

        for train_index, test_index in self.cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            fit_args = self.get_pipeline_args(X_train, y_train, data, time_index, index)
            pipeline = self.clone_pipeline(pipeline)
            pipeline.fit(**fit_args)

            predict_args = self.get_pipeline_args(X_test, None, data, time_index, index)
            predictions = pipeline.predict(**predict_args)
            score = self.scorer(y_test, predictions)
            scores.append(score)

        return np.mean(scores)

    def to_dicts(self, hyperparameters):

        params_tree = defaultdict(dict)
        for (block, hyperparameter), value in hyperparameters.items():
            if isinstance(value, np.integer):
                value = int(value)

            elif isinstance(value, np.floating):
                value = float(value)

            elif isinstance(value, np.ndarray):
                value = value.tolist()

            elif value == 'None':
                value = None

            params_tree[block][hyperparameter] = value

        return params_tree

    def to_tuples(self, params_tree, tunable_keys):
        param_tuples = defaultdict(dict)
        for block_name, params in params_tree.items():
            for param, value in params.items():
                key = (block_name, param)
                if key in tunable_keys:
                    param_tuples[key] = 'None' if value is None else value

        return param_tuples

    @staticmethod
    def get_pipeline_args(X, y, data, time_index, index):
        """Build a dictionary with all the arguments needed to call the pipeline.

        Args:
            X(pd.DataFrame):
            y:
            data:

        """

        entities = {
            'main': (X, index, None),
            'timeseries': (data, time_index, 'time')
        }

        relationships = [('main', index, 'timeseries', index)]

        pipeline_args = {
            'X': X,
            'entities': entities,
            'relationships': relationships,
            'target_entity': 'main'
        }
        if y is not None:
            pipeline_args['y'] = y

        return pipeline_args

    def tune(self, X, y, data, time_index, index, iterations=10):

        # Build the tuner passing the tunables for
        # this pipeline
        if not self.tuner:
            tunables, tunable_keys = self.get_tunables()
            self.tuner = GP(tunables)

            # Compute an initial best score
            best_score = self.score_pipeline(self.pipeline, X, y, data, time_index, index)

            LOGGER.info("Default hyperparameters score: {}".format(best_score))

            # Inform the tuner about the score that the default hyperparmeters obtained
            param_tuples = self.to_tuples(self.pipeline.get_hyperparameters(), tunable_keys)
            self.tuner.add(param_tuples, best_score)

        for i in range(iterations):
            LOGGER.info("Scoring pipeline {}".format(i + 1))

            # Get hyperparameters proposal
            params = self.tuner.propose(1)

            # Convert hyperparameters to tree of dicts as expected by the MLPipeline
            param_dicts = self.to_dicts(params)

            # Get a new instance of the pipeline with the same primitives
            candidate = self.clone_pipeline(self.pipeline)

            # Set the new hyperparameters to the cloned pipeline and calculate their score
            candidate.set_hyperparameters(param_dicts)

            score = self.score_pipeline(candidate, X, y, data, time_index, index)

            # Inform the tuner about the obtained score
            self.tuner.add(params, score)

            # See if this pipeline is the best one so far
            if self._is_better(score, best_score):
                best_score = score
                self.set_hyperparameters(param_dicts)

    def fit(self, X, y, data, time_index, index):
        pipeline_args = self.get_pipeline_args(X, y, data, time_index, index)
        self.pipeline.fit(**pipeline_args)
        self.fitted = True

    def predict(self, X, data, time_index, index):
        if not self.fitted:
            raise NotFittedError()

        pipeline_args = self.get_pipeline_args(X, None, data, time_index, index)

        return self.pipeline.predict(**pipeline_args)


class TimeSeriesClassifier(TimeSeriesEstimator):
    maximize_score = True
    cv_class = StratifiedKFold
    template = {
        'primitives': [
            'mlprimitives.preprocessing.ClassEncoder',
            'featuretools.dfs',
            'sklearn.preprocessing.Imputer',
            'xgboost.XGBClassifier',
            'mlprimitives.preprocessing.ClassDecoder'
        ],
        'hyperparameters': {
            'xgboost.XGBClassifier#1': {
                'learning_rate': 0.1,
                'n_estimators': 300,
                'max_depth': 3,
                'gamma': 0,
                'min_child_weight': 1
            }
        },
        'init_params': {
            'featuretools.dfs#1': {
                'encode': True
            }
        }
    }


class TimeSeriesRegressor(TimeSeriesEstimator):
    maximize_score = False
    cv_class = KFold
    template = {
        'primitives': [
            'featuretools.dfs',
            'sklearn.preprocessing.Imputer',
            'sklearn.preprocessing.StandardScaler',
            'xgboost.XGBRegressor'
        ],
        'hyperparameters': {
            'xgboost.XGBRegressor#1': {
                'n_jobs': -1,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'max_depth': 3,
                'gamma': 0,
                'min_child_weight': 1
            }
        },
        'init_params': {
            'featuretools.dfs#1': {
                'encode': True
            }
        }
    }
