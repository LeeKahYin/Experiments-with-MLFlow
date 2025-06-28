# Experiments-with-MLFlow
MLflow is a powerful tool for tracking and managing machine learning experiments. 

---

## ✅ What Can Be Tracked Using MLflow

### 1. **Metrics**
- **Accuracy**: Track model accuracy over different runs.
- **Loss**: Log training and validation loss during the training process.
- **Precision, Recall, F1-Score**: Useful for classification tasks.
- **AUC (Area Under Curve)**: For evaluating binary classifiers.
- **Custom Metrics**: Any numeric value such as RMSE, MAE, etc.

### 2. **Parameters**
- **Model Hyperparameters**: E.g., `learning_rate`, `n_estimators`, `max_depth`.
- **Data Processing Parameters**: E.g., train-test split ratio, feature scaling.
- **Feature Engineering**: Parameters related to transformations or extractions.

### 3. **Artifacts**
- **Trained Models**: Save and version models.
- **Model Summaries**: Architecture, parameter counts.
- **Confusion Matrices**: For evaluating classification performance.
- **ROC Curves**: Visual ROC plots.
- **Custom Plots**: Loss curves, feature importances, etc.
- **Input Data**: Training/testing datasets.
- **Scripts & Notebooks**: Log source files used.
- **Environment Files**: `requirements.txt`, `conda.yaml`, etc.

### 4. **Models**
- **Pickled Models**: `.pkl` files that can be reloaded.
- **ONNX Models**: For cross-platform deployment.
- **Custom Models**: Via MLflow’s model interface.

### 5. **Tags**
- **Run Tags**: Metadata like author, experiment name, model type.
- **Environment Tags**: E.g., `gpu`, `cloud_provider`.

### 6. **Source Code**
- **Scripts**: Training scripts or Jupyter notebooks.
- **Git Commit**: Commit hash to link with a specific code version.
- **Dependencies**: Library versions and runtime environment.

### 7. **Logging Inputs and Outputs**
- **Training Data**: Data used for training.
- **Test Data**: Data used for validation/testing.
- **Inference Outputs**: Model predictions and outputs.

### 8. **Custom Logging**
- **Custom Objects**: Any Python object or custom artifact.
- **Custom Functions**: Log details of any custom pipeline or method.

### 9. **Model Registry**
- **Model Versioning**: Track multiple versions and their lifecycle stages (`Staging`, `Production`, etc.).
- **Model Deployment**: Manage deployment status and metadata.

### 10. **Run and Experiment Details**
- **Run ID**: Unique identifier for each run.
- **Experiment Name**: Group multiple related runs.
- **Timestamps**: Start/end times of training.

---

## ⚙️ `mlflow.autolog()` Overview

`mlflow.autolog()` is a convenient utility to automatically log standard ML metadata.

### ✅ What Is Logged Automatically

1. **Parameters**
   - E.g., `max_depth`, `learning_rate`, `n_estimators`.

2. **Metrics**
   - Accuracy, precision, recall, loss, etc. (depending on framework).

3. **Model**
   - Automatically logs the trained model.

4. **Artifacts**
   - Some default plots and summaries.

5. **Framework-Specific Details**
   - E.g., early stopping parameters, epochs, optimizers.

6. **Environment Info**
   - Installed packages and versions.

7. **Training Data Info**
   - Size and general info (not full dataset).

8. **Model Signature**
   - Automatically infers input/output schema.

---

### ❌ What Is *Not* Logged Automatically

1. **Custom Metrics**
   - E.g., F1 Score, RMSE, etc. (must be logged manually).

2. **Custom Artifacts**
   - Custom plots, reports, or visualizations.

3. **Preprocessed Data**
   - Input/output data must be logged manually if needed.

4. **Intermediate Model States**
   - Not captured unless logged explicitly.

5. **Complex Model Structures**
   - Highly customized or non-standard models may not be logged correctly.

6. **Custom Training Loops**
   - Not supported if outside the framework’s standard training procedure.

7. **Unsupported Frameworks**
   - Autologging won’t work if the framework is not supported.

8. **Custom Hyperparameter Tuning**
   - Grid search or special tuning outside the expected pattern.


