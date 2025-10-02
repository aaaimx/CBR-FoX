"""
Minimal test suite for cbr_fox package.
Ensures basic functionality and Docker container compatibility.
"""

import pytest
import numpy as np


def test_imports():
    """Test that all main modules can be imported."""
    from cbr_fox.core import cbr_fox
    from cbr_fox.builder import cbr_fox_builder
    from cbr_fox.adapters import sktime_interface
    from cbr_fox.custom_distance import cci_distance
    assert True


def test_cbr_fox_initialization():
    """Test basic cbr_fox initialization."""
    from cbr_fox.core import cbr_fox

    cbr = cbr_fox(metric="euclidean")
    assert cbr.metric == "euclidean"
    assert cbr.smoothness_factor == 0.2


def test_cbr_fox_basic_workflow():
    """Test basic fit-predict workflow."""
    from cbr_fox.core import cbr_fox

    np.random.seed(42)

    # Create minimal sample data
    training_windows = np.random.randn(30, 10, 2)
    target_training_windows = np.random.randn(30, 2)
    forecasted_window = np.random.randn(10, 2)
    prediction = np.random.randn(2)

    # Initialize and run
    cbr = cbr_fox(metric="euclidean")
    cbr.fit(training_windows, target_training_windows, forecasted_window)
    cbr.predict(prediction, num_cases=3, mode="simple")

    # Verify results exist
    report = cbr.get_analysis_report()
    assert report is not None
    assert len(report) > 0


def test_cbr_fox_builder_initialization():
    """Test cbr_fox_builder with multiple techniques."""
    from cbr_fox.core import cbr_fox
    from cbr_fox.builder import cbr_fox_builder

    techniques = [
        cbr_fox(metric="euclidean"),
        cbr_fox(metric="dtw"),
    ]

    builder = cbr_fox_builder(techniques)
    assert len(builder.techniques_dict) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])