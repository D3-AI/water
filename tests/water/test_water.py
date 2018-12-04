#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `water` package."""
import warnings
from unittest import TestCase

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

    def test_tune(self):
        """tune select the best hyperparameters for the given data."""
        # Setup
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

        scorer = f1_score
        cv = StratifiedKFold(n_splits=2, shuffle=True)
        instance = TimeSeriesClassifier(cv=cv, scorer=scorer)
        initial_score = instance.score_pipeline(instance.pipeline, X, y, data, time_index, index)

        # Run
        instance.tune(X, y, data, time_index, index, iterations=1)
        final_score = instance.score_pipeline(instance.pipeline, X, y, data, time_index, index)

        # Check
        assert instance.tuner is not None
        assert instance.fitted is False
        assert initial_score <= final_score

    def test_fit(self):
        """fit prepare the pipeline to make predictions based on the given data."""
        # Setup
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

        scorer = f1_score
        cv = StratifiedKFold(n_splits=2, shuffle=True)
        instance = TimeSeriesClassifier(cv=cv, scorer=scorer)

        # Run
        instance.fit(X, y, data, time_index, index)

        # Check
        assert instance.fitted

    def test_predict(self):
        """fit prepare the pipeline to make predictions based on the given data."""
        # Setup
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

        scorer = f1_score
        cv = StratifiedKFold(n_splits=2, shuffle=True)
        instance = TimeSeriesClassifier(cv=cv, scorer=scorer)
        instance.fit(X, y, data, time_index, index)
        # Run
        predictions = instance.predict(X, data, time_index, index)

        # Check
        assert instance.fitted
        assert predictions.shape == y.shape
