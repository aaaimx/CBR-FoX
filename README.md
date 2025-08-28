# CBR-FoX: Case-Based Reasoning for Time Series Prediction Explanations

CBR-FoX is a Python library designed to provide case-based reasoning explanations for time series prediction models. This approach enhances the transparency and interpretability of machine learning models applied to sequential data.

## Features

- **Case-Based Reasoning (CBR) Implementation:** Utilizes case-based reasoning to enhance explainability in time series predictions.
- **Versatile & Adaptable:** Supports various types of time series data.
- **ML Model Compatibility:** Easily integrates with common machine learning models.
- **Comprehensible Explanations:** Provides clear, human-readable insights into model predictions.

## Installation

To install CBR-FoX and its dependencies, either clone the repository or use the pip package manager:

### Install via PyPI
```bash
pip install CBR-FoX
```

### Install via GitHub
```bash
# Clone the repository
git clone https://github.com/aaaimx/CBR-FoX.git
cd CBR-FoX

# Install required dependencies
pip install -r requirements.txt
```

## Usage

Follow these steps to use CBR-FoX in your projects:

### 1. Retrieve Model Information
Extract the relevant inputs and outputs from your AI model.

### 2. Create CBR-FoX Instances
```python
from cbr_fox import CBRfoxInstances
cbr_instances = CBRfoxInstances(model_outputs)
```

### 3. Initialize the Builder
```python
from cbr_fox import CBRfoxBuilder
builder = CBRfoxBuilder(cbr_instances)
```

### 4. Train the Instance
```python
builder.fit(train_windows, train_targets, target_to_analyze, window_to_predict)
```

### 5. Obtain Explanations
```python
builder.predict(prediction=prediction, num_cases=5)
```

### 6. Visualize Results
```python
builder.visualize_pyplot(
    fmt='--d',
    scatter_params={'s': 50},
    xtick_rotation=50,
    title='Example Visualization',
    xlabel='X-axis',
    ylabel='Y-axis'
)
```

## Library Usage Diagram

The following diagram illustrates the typical workflow of CBR-FoX, from retrieving AI model outputs to generating visual explanations.

![Library Basic Usage Diagram](https://github.com/aaaimx/CBR-FoX/blob/develop/library_basic_usage_diagram.svg)

## Library File Relation Diagram

The following diagram represents the core classes and their interactions within the library. The `cci_distance` file is utilized when an instance is created using the custom distance metric implemented in this script.

![Library File Relation Diagram](https://github.com/aaaimx/CBR-FoX/blob/develop/file_relation_diagram.svg)

---
For further details, check out the official documentation: [CBR-FoX on Read the Docs](https://cbr-fox.readthedocs.io/en/latest/overview.html).





