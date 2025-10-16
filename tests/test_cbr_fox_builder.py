"""
Comprehensive unit tests for cbr_fox_builder module.
Tests builder pattern, multiple techniques management, and visualization.
"""

import pytest
import numpy as np
from cbr_fox.core import cbr_fox
from cbr_fox.builder import cbr_fox_builder
from unittest.mock import patch
import io
import sys


class Test_builder_initialization:
    """Test builder initialization with different configurations."""

    def test_empty_techniques_list(self):
        """Test with empty techniques list."""
        with patch('sys.stdout', new=io.StringIO()):  # Suppress debug prints
            builder = cbr_fox_builder([])
            assert len(builder.techniques_dict) == 0

    def test_single_technique(self):
        """Test with single technique."""
        with patch('sys.stdout', new=io.StringIO()):
            techniques = [cbr_fox(metric="euclidean")]
            builder = cbr_fox_builder(techniques)
            assert len(builder.techniques_dict) == 1
            assert "euclidean" in builder.techniques_dict

    def test_multiple_techniques_string_metrics(self):
        """Test with multiple techniques using string metrics."""
        with patch('sys.stdout', new=io.StringIO()):
            techniques = [
                cbr_fox(metric="euclidean"),
                cbr_fox(metric="dtw"),
                cbr_fox(metric="squared")
            ]
            builder = cbr_fox_builder(techniques)
            assert len(builder.techniques_dict) == 3
            assert "euclidean" in builder.techniques_dict
            assert "dtw" in builder.techniques_dict
            assert "squared" in builder.techniques_dict

    def test_techniques_with_callable_metrics(self):
        """Test with callable metrics."""
        def custom_metric_1(input_dict, **kwargs):
            return np.random.rand(len(input_dict['training_windows']), 1)

        def custom_metric_2(input_dict, **kwargs):
            return np.random.rand(len(input_dict['training_windows']), 1)

        with patch('sys.stdout', new=io.StringIO()):
            techniques = [
                cbr_fox(metric=custom_metric_1),
                cbr_fox(metric=custom_metric_2)
            ]
            builder = cbr_fox_builder(techniques)
            assert len(builder.techniques_dict) == 2
            assert "custom_metric_1" in builder.techniques_dict
            assert "custom_metric_2" in builder.techniques_dict

    def test_mixed_metric_types(self):
        """Test with mixed string and callable metrics."""
        def custom_metric(input_dict, **kwargs):
            return np.random.rand(len(input_dict['training_windows']), 1)

        with patch('sys.stdout', new=io.StringIO()):
            techniques = [
                cbr_fox(metric="euclidean"),
                cbr_fox(metric=custom_metric),
                cbr_fox(metric="dtw")
            ]
            builder = cbr_fox_builder(techniques)
            assert len(builder.techniques_dict) == 3


class Test_builder_fit_method:
    """Test builder fit method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        training_windows = np.random.randn(50, 24, 3)
        target_training_windows = np.random.randn(50, 3)
        forecasted_window = np.random.randn(24, 3)
        return training_windows, target_training_windows, forecasted_window

    @pytest.fixture
    def builder_with_techniques(self):
        """Create builder with multiple techniques."""
        with patch('sys.stdout', new=io.StringIO()):
            techniques = [
                cbr_fox(metric="euclidean"),
                cbr_fox(metric="dtw")
            ]
            return cbr_fox_builder(techniques)

    def test_fit_all_techniques(self, builder_with_techniques, sample_data):
        """Test that fit is applied to all techniques."""
        train_w, target_w, forecast_w = sample_data
        builder_with_techniques.fit(train_w, target_w, forecast_w)

        for name in builder_with_techniques.techniques_dict:
            technique = builder_with_techniques.techniques_dict[name]
            assert technique.correlation_per_window is not None
            assert technique.smoothed_correlation is not None

    def test_fit_creates_internal_structures(self, builder_with_techniques, sample_data):
        """Test that fit creates internal structures for all techniques."""
        train_w, target_w, forecast_w = sample_data
        builder_with_techniques.fit(train_w, target_w, forecast_w)

        for name in builder_with_techniques.techniques_dict:
            technique = builder_with_techniques.techniques_dict[name]
            assert technique.input_data_dictionary is not None
            assert len(technique.best_windows_index) > 0


class Test_builder_predict_method:
    """Test builder predict method."""

    @pytest.fixture
    def fitted_builder(self):
        """Create a fitted builder."""
        np.random.seed(42)
        training_windows = np.random.randn(50, 24, 3)
        target_training_windows = np.random.randn(50, 3)
        forecasted_window = np.random.randn(24, 3)

        with patch('sys.stdout', new=io.StringIO()):
            techniques = [
                cbr_fox(metric="euclidean"),
                cbr_fox(metric="dtw")
            ]
            builder = cbr_fox_builder(techniques)
            builder.fit(training_windows, target_training_windows, forecasted_window)
            return builder

    def test_predict_all_techniques(self, fitted_builder):
        """Test that predict is applied to all techniques."""
        prediction = np.random.randn(3)
        fitted_builder.predict(prediction, num_cases=5, mode="simple")

        for name in fitted_builder.techniques_dict:
            technique = fitted_builder.techniques_dict[name]
            assert technique.analysis_report is not None
            assert technique.analysis_report_combined is not None

    def test_predict_different_num_cases(self, fitted_builder):
        """Test predict with different number of cases."""
        prediction = np.random.randn(3)

        for num_cases in [3, 5, 10]:
            fitted_builder.predict(prediction, num_cases=num_cases, mode="simple")

            for name in fitted_builder.techniques_dict:
                technique = fitted_builder.techniques_dict[name]
                report = technique.get_analysis_report()
                assert len(report) <= num_cases * 2

    def test_predict_simple_mode(self, fitted_builder):
        """Test predict with simple mode."""
        prediction = np.random.randn(3)
        fitted_builder.predict(prediction, num_cases=5, mode="simple")

        for name in fitted_builder.techniques_dict:
            technique = fitted_builder.techniques_dict[name]
            assert technique.analysis_report is not None

    def test_predict_weighted_mode(self, fitted_builder):
        """Test predict with weighted mode."""
        prediction = np.random.randn(3)
        fitted_builder.predict(prediction, num_cases=5, mode="weighted")

        for name in fitted_builder.techniques_dict:
            technique = fitted_builder.techniques_dict[name]
            assert technique.analysis_report is not None


class Test_builder_dictionary_access:
    """Test dictionary-like access to techniques."""

    @pytest.fixture
    def builder(self):
        """Create builder with techniques."""
        with patch('sys.stdout', new=io.StringIO()):
            techniques = [
                cbr_fox(metric="euclidean"),
                cbr_fox(metric="dtw"),
                cbr_fox(metric="squared")
            ]
            return cbr_fox_builder(techniques)

    def test_getitem_valid_technique(self, builder):
        """Test accessing valid technique."""
        technique = builder["euclidean"]
        assert technique is not None
        assert technique.metric == "euclidean"

    def test_getitem_all_techniques(self, builder):
        """Test accessing all techniques."""
        for metric_name in ["euclidean", "dtw", "squared"]:
            technique = builder[metric_name]
            assert technique.metric == metric_name

    def test_getitem_invalid_technique(self, builder):
        """Test accessing non-existent technique."""
        with pytest.raises(KeyError):
            _ = builder["nonexistent_metric"]

    def test_getitem_returns_correct_instance(self, builder):
        """Test that getitem returns the correct instance."""
        technique = builder["euclidean"]
        assert isinstance(technique, cbr_fox)
        assert technique.metric == "euclidean"

class Test_builder_workflow:
    """Test complete builder workflow."""

    def test_complete_workflow(self):
        """Test complete fit-predict-visualize workflow."""
        np.random.seed(42)
        training_windows = np.random.randn(50, 24, 3)
        target_training_windows = np.random