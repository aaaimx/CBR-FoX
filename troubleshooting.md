# Troubleshooting

This section covers common issues and their solutions when working with CBR-FoX.

## Installation Issues

### 1. Dependency Conflicts

**Problem**: Conflicting package versions during installation.

```bash
ERROR: Could not find a version that satisfies the requirement sktime==0.29.0
```

**Solution**:
- Check your Python version (requires Python 3.8+)
- Update pip to the latest version:
  ```bash
  pip install --upgrade pip
  ```
- Install with specific version constraints:
  ```bash
  pip install cbr-fox --no-deps
  pip install -r requirements.txt
  ```

### 2. NumPy Compatibility Issues

**Problem**: NumPy version incompatibility errors.

```bash
ImportError: numpy.dtype size changed, may indicate binary incompatibility
```

**Solution**:
- Reinstall NumPy:
  ```bash
  pip uninstall numpy
  pip install numpy==1.26.4
  ```
- Use virtual environment to avoid conflicts:
  ```bash
  python -m venv cbr_fox_env
  source cbr_fox_env/bin/activate  # On Windows: cbr_fox_env\Scripts\activate
  pip install cbr-fox
  ```

### 3. sktime Installation Issues

**Problem**: sktime dependency fails to install properly.

```bash
ERROR: Failed building wheel for sktime
```

**Solution**:
- Install pre-compiled binaries:
  ```bash
  pip install --only-binary=all sktime==0.29.0
  ```
- For development environments:
  ```bash
  conda install -c conda-forge sktime
  ```

## Runtime Errors

### 4. Memory Issues with Large Time Series

**Problem**: Out of memory errors when processing large datasets.

```python
MemoryError: Unable to allocate array
```

**Solution**:
- Reduce the number of cases:
  ```python
  # Reduce num_cases parameter
  p.predict(prediction=prediction, num_cases=3)  # Instead of 10+
  ```
- Process data in chunks:
  ```python
  # Process smaller windows
  window_size = 7  # Instead of 14 or larger
  ```
- Use data subsampling:
  ```python
  # Subsample your time series
  data_subset = training_windows[::2]  # Take every 2nd sample
  ```

### 5. Similarity Metric Errors

**Problem**: Custom similarity metric raises errors.

```python
TypeError: 'NoneType' object is not callable
```

**Solution**:
- Ensure proper metric implementation:
  ```python
  from custom_distance.cci_distance import cci_distance
  
  # Correct initialization
  cbr_fox_instance = cbr_fox.cbr_fox(
      metric=cci_distance, 
      kwargs={"punishedSumFactor": 0.5}
  )
  ```
- Check metric compatibility with sktime:
  ```python
  # Use built-in metrics if custom ones fail
  from sktime.distances import dtw_distance
  cbr_fox_instance = cbr_fox.cbr_fox(metric=dtw_distance)
  ```

### 6. Input Data Format Issues

**Problem**: Incorrect data shape or type errors.

```python
ValueError: Input arrays must have the same shape
```

**Solution**:
- Verify data dimensions:
  ```python
  print(f"Training windows shape: {training_windows.shape}")
  print(f"Target windows shape: {target_training_windows.shape}")
  print(f"Forecasted window shape: {forecasted_window.shape}")
  
  # Ensure proper reshaping
  target_training_windows = target_training_windows.reshape(-1, 1)
  ```
- Check data types:
  ```python
  # Convert to numpy arrays if needed
  import numpy as np
  training_windows = np.array(training_windows, dtype=np.float64)
  ```

## Visualization Problems

### 7. Matplotlib Display Issues

**Problem**: Plots not showing or displaying incorrectly.

```python
# No plot appears
p.visualize_pyplot()
```

**Solution**:
- Enable interactive plotting:
  ```python
  import matplotlib.pyplot as plt
  plt.ion()  # Turn on interactive mode
  
  # After visualization
  plt.show()
  ```
- Check backend configuration:
  ```python
  import matplotlib
  matplotlib.use('Agg')  # For headless environments
  # or
  matplotlib.use('TkAgg')  # For GUI environments
  ```

### 8. Unicode/Character Encoding Issues

**Problem**: Special characters in plot labels cause errors.

```python
UnicodeDecodeError: 'ascii' codec can't decode byte
```

**Solution**:
- Use UTF-8 encoding:
  ```python
  p.visualize_pyplot(
      title="Power Usage Predictions",  # Avoid special characters
      xlabel="Time (Months)",
      ylabel="Power Usage (kWh)"
  )
  ```

## Performance Issues

### 9. Slow Processing with Large Datasets

**Problem**: CBR-FoX takes too long to process.

**Solution**:
- Optimize smoothing parameters:
  ```python
  # Reduce smoothness factor
  cbr_fox_instance = cbr_fox.cbr_fox(
      metric=cci_distance,
      kwargs={"smoothness_factor": 0.1}  # Lower value for faster processing
  )
  ```
- Use parallel processing (if available):
  ```python
  # Process smaller batches
  batch_size = 1000
  for i in range(0, len(training_windows), batch_size):
      batch = training_windows[i:i+batch_size]
      # Process batch
  ```

### 10. High Memory Usage

**Problem**: Memory consumption grows excessively.

**Solution**:
- Clear intermediate results:
  ```python
  import gc
  
  # After processing
  del training_windows
  gc.collect()
  ```
- Use memory-efficient data types:
  ```python
  # Use float32 instead of float64 if precision allows
  training_windows = training_windows.astype(np.float32)
  ```

## API and Usage Issues

### 11. Incorrect Parameter Types

**Problem**: Type errors when calling methods.

```python
TypeError: Expected list, got numpy.ndarray
```

**Solution**:
- Convert data types explicitly:
  ```python
  # Convert to appropriate types
  techniques = [
      cbr_fox.cbr_fox(metric=cci_distance, kwargs={"punishedSumFactor": float(0.5)})
  ]
  ```

### 12. Missing Required Parameters

**Problem**: Essential parameters not provided.

```python
TypeError: missing required argument
```

**Solution**:
- Check required parameters:
  ```python
  # Ensure all required parameters are provided
  p = cbr_fox_builder(techniques)
  p.fit(
      training_windows=training_windows,
      target_training_windows=target_training_windows.reshape(-1, 1),
      forecasted_window=forecasted_window
  )
  ```

## Environment-Specific Issues

### 13. Jupyter Notebook Issues

**Problem**: Plots not displaying in Jupyter notebooks.

**Solution**:
- Use inline plotting:
  ```python
  %matplotlib inline
  import matplotlib.pyplot as plt
  ```

### 14. Windows Path Issues

**Problem**: File path errors on Windows systems.

**Solution**:
- Use raw strings or forward slashes:
  ```python
  # Instead of: data = np.load("data\file.npz")
  data = np.load("data/file.npz")
  # or
  data = np.load(r"data\file.npz")
  ```

## Getting Help

If you encounter issues not covered here:

1. **Check the documentation**: Visit [CBR-FoX Documentation](https://cbr-fox.readthedocs.io/)
2. **Search existing issues**: Check the [GitHub Issues](https://github.com/aaaimx/CBRFoX/issues)
3. **Create a new issue**: If your problem isn't documented, create a new issue with:
   - Python version
   - CBR-FoX version
   - Complete error traceback
   - Minimal code example to reproduce the issue
   - Operating system information

4. **Contact support**: Email support@aaaimx.org for additional assistance

## Debug Mode

Enable debug mode for more detailed error information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your CBR-FoX code here
```

This will provide more detailed information about what's happening during execution and can help identify the source of issues.