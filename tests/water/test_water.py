#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `water` package."""
import warnings
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from water.water import TimeSeriesClassifier

warnings.filterwarnings('ignore', category=DeprecationWarning)


class TestTimeSeriesClassifier(TestCase):
    """Tests for `water` package."""

    def test___init___default(self):
        """If cv_splits is passed a new cv object is created with the specified param."""
        # Setup
        scorer = f1_score

        # Run
        instance = TimeSeriesClassifier(cv_splits=5, scorer=scorer)

        # Check
        assert instance.cv.n_splits == 5
        assert instance.scorer == scorer
        assert instance.fitted is False
        assert instance.tuner is None

    def test___init___cv_object(self):
        """If a cv object is passed as cv argument, it will be used for cross-validation."""
        # Setup
        scorer = f1_score
        cv = StratifiedKFold(n_splits=5, shuffle=True)

        # Run
        instance = TimeSeriesClassifier(cv=cv, scorer=scorer)

        # Check
        assert instance.cv == cv
        assert instance.scorer == scorer
        assert instance.fitted is False
        assert instance.tuner is None

    def test___init___no_cv_raises_exception(self):
        """If no cv or cv_splits argument is passed, a ValueError is raised."""
        # Setup
        scorer = f1_score

        # Run/Check
        with self.assertRaises(ValueError):
            TimeSeriesClassifier(scorer)

    def test_get_pipeline_args(self):
        """get_pipeline_arg return the arguments needed for tune, fit and predict."""
        # Setup
        time_index = 'timeseries_id'
        index = 'D3MIndex'

        X = pd.DataFrame({
            index: [0, 1, 2]
        })
        X.set_index(X[index])

        y = pd.Series([0, 1, 0], name=index)
        y.set_index = y

        data = pd.DataFrame({
            index: [0, 0, 0, 1, 1, 1, 2, 2, 2],
            'time': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            time_index: [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'value': [0, 1, 0, 1, 0, 1, 0, 1, 0]
        })

        scorer = f1_score
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        instance = TimeSeriesClassifier(cv=cv, scorer=scorer)

        expected_entities = {
            'main': (X, index, None),
            'timeseries': (data, time_index, 'time')
        }

        expected_relationships = [
            ('main', index, 'timeseries', index)
        ]
        expected_target_entity = 'main'

        # Run
        result = instance.get_pipeline_args(X, y, data, time_index, index)

        # Check
        assert result['X'].equals(X)
        assert result['y'].equals(y)
        assert result['target_entity'] == expected_target_entity
        assert result['entities'] == expected_entities
        assert result['relationships'] == expected_relationships

    @patch('water.water.GP')
    def test_tune(self, gp_mock):
        """tune select the best hyperparameters for the given data."""
        # Setup - Data
        time_index = 'timeseries_id'
        index = 'D3MIndex'

        X = pd.DataFrame({
            index: [0, 1, 2, 3, 4, 5]
        })
        X.set_index(X[index])

        y = pd.Series([0, 1, 0, 1, 0, 1], name=index)
        y.set_index = y

        data = pd.DataFrame({
            index: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            'time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            time_index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            'value': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

        # Setup - Classifier
        iterations = 2
        scorer = f1_score
        cv = StratifiedKFold(n_splits=2, shuffle=True)
        instance = TimeSeriesClassifier(cv=cv, scorer=scorer)
        tunables, tunable_keys = instance.get_tunables()

        # Setup - Mock
        gp_mock_instance = MagicMock()
        gp_mock.return_value = gp_mock_instance
        expected_propose_calls = [((1, ), ), ((1, ), )]
        expected_best_score = 0.0
        param_tuples = instance.to_tuples(instance.pipeline.get_hyperparameters(), tunable_keys)
        expected_add_calls = [
            ((param_tuples, expected_best_score), ),
            ((gp_mock_instance.propose.return_value, expected_best_score), ),
            ((gp_mock_instance.propose.return_value, expected_best_score), ),
        ]

        # Run
        instance.tune(X, y, data, time_index, index, iterations=iterations)

        # Check
        gp_mock.assert_called_once_with(tunables)
        assert instance.tuner == gp_mock_instance

        assert gp_mock_instance.propose.call_count == iterations
        assert gp_mock_instance.propose.call_args_list == expected_propose_calls

        assert gp_mock_instance.add.call_count == iterations + 1
        assert gp_mock_instance.add.call_args_list == expected_add_calls

        assert instance.fitted is False

    @patch('water.water.MLPipeline')
    def test_fit(self, pipeline_mock):
        """fit prepare the pipeline to make predictions based on the given data."""
        # Setup - Data
        time_index = 'timeseries_id'
        index = 'D3MIndex'

        X = pd.DataFrame({
            index: [0, 1, 2, 3, 4, 5]
        })
        X.set_index(X[index])

        y = pd.Series([0, 1, 0, 1, 0, 1], name=index)
        y.set_index = y

        data = pd.DataFrame({
            index: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            'time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            time_index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            'value': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

        # Setup - Mock
        pipeline_mock_instance = MagicMock()
        pipeline_mock.from_dict.return_value = pipeline_mock_instance

        # Setup - Classifier
        scorer = f1_score
        cv = StratifiedKFold(n_splits=2, shuffle=True)
        instance = TimeSeriesClassifier(cv=cv, scorer=scorer)

        # Setup - Expected results
        expected_fit_args = instance.get_pipeline_args(X, y, data, time_index, index)

        # Run
        instance.fit(X, y, data, time_index, index)

        # Check
        pipeline_mock.from_dict.assert_called_once_with(instance.template)
        assert instance.pipeline == pipeline_mock_instance

        pipeline_mock_instance.fit.assert_called_once_with(**expected_fit_args)
        assert instance.fitted

    @patch('water.water.MLPipeline')
    def test_predict(self, pipeline_mock):
        """predict produces results using the pipeline."""
        # Setup - Data
        time_index = 'timeseries_id'
        index = 'D3MIndex'

        X = pd.DataFrame({
            index: [0, 1, 2, 3, 4, 5]
        })
        X.set_index(X[index])

        y = pd.Series([0, 1, 0, 1, 0, 1], name=index)
        y.set_index = y

        data = pd.DataFrame({
            index: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            'time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            time_index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            'value': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

        # Setup - Mock
        pipeline_mock_instance = MagicMock()
        pipeline_mock.from_dict.return_value = pipeline_mock_instance

        # Setup - Classifier
        scorer = f1_score
        cv = StratifiedKFold(n_splits=2, shuffle=True)
        instance = TimeSeriesClassifier(cv=cv, scorer=scorer)
        instance.fit(X, y, data, time_index, index)

        # Setup - Expected results
        expected_fit_args = instance.get_pipeline_args(X, y, data, time_index, index)
        expected_predict_args = instance.get_pipeline_args(X, None, data, time_index, index)

        # Run
        instance.predict(X, data, time_index, index)

        # Check
        pipeline_mock.from_dict.assert_called_once_with(instance.template)
        assert instance.pipeline == pipeline_mock_instance

        pipeline_mock_instance.fit.assert_called_once_with(**expected_fit_args)
        assert instance.fitted

        pipeline_mock_instance.predict.assert_called_once_with(**expected_predict_args)
