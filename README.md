# 🧪 MLflow Experiment Tracking Exploration

This repository is a hands-on guide to using **MLflow** for experiment tracking, including examples of:

- Local and remote tracking
- Autologging
- Hyperparameter tuning and parent-child runs
- Integration with [DagsHub](https://dagshub.com)

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `src/local_tracking.py` | Tracks experiments **locally** using `mlflow ui`. |
| `src/remote_tracking.py` | Demonstrates how to use **DagsHub** as a remote tracking server. |
| `src/autolog.py` | Uses **MLflow autologging** to automatically log models, metrics, and params. |
| `src/hyperparameter_tuning.py` | Logs **parent and child runs** for hyperparameter tuning and compares the metrics in MLflow UI. |

---

## 🔧 Setup Instructions

### 1. Install Required Packages

> ✅ Note: `mlflow==2.8.1` is used for compatibility with DagsHub.

```bash
pip install mlflow==2.8.1 scikit-learn dagshub
```
---

## 🖥️ Local Experiment Tracking (`local_tracking.py`)

This script shows how to track your experiments **locally**:

### 🔹 Steps:
1. Start the MLflow UI in your terminal:

```bash
mlflow ui
```

2. Obtain the tracking URI (result of your previous run, usually `http://127.0.0.1:5000`) and update your script at line 10:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

3. Run `python local_tracking.py` and view your runs on the MLflow UI in your browser.

---

## 🌐 Remote Tracking via DagsHub (`remote_tracking.py`)

This script shows how to use **DagsHub** as your MLflow tracking server.

### 🔹 Steps to Connect Git Repo to DagsHub:
1. Create an account and create a new repo on [dagshub.com](https://dagshub.com).
2. Connect your repo to your github repo.
3. Navigate to **Remote > Experiments** in your DagsHub repo.
4. Click **go to mlflow UI** for the webpage.
5. Copy the ```import dagshub and dagshub.init(...) ```

![image](https://github.com/user-attachments/assets/7d0b5ee7-90be-49ff-bc18-c0b5519393e3)


### 🔹 Update Your Script at line 9-13:

```python
import dagshub
dagshub.init(repo_owner='your_name', repo_name='your_repo_name', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/your_mlflow_UI_URL")
```

> ℹ️ **Note:** DagsHub currently supports MLflow **up to v2.8.1**, so I’ve pinned this version.

---

## ⚙️ MLflow Autologging (`autolog.py`)

This script demonstrates the use of MLflow’s **autologging** feature.

```python
import mlflow.sklearn
mlflow.sklearn.autolog()
```

Once enabled, this will automatically log:

- Estimator parameters
- Evaluation metrics
- Model artifacts
- Feature importance (if applicable)

---

## 🔁 Hyperparameter Tuning with Parent-Child Runs (`hyperparameter_tuning.py`)

This script shows how to structure **nested runs** (child runs) under a **parent run** when tuning hyperparameters.

### 🧪 Highlights:
- Organize each hyperparameter combination as a child run.
- Automatically record metrics and params for each trial.
- Use MLflow UI to **compare runs** and **identify the best performing one**.

```python
   with mlflow.start_run() as parent:
       grid_search.fit(X_train, y_train)
   
       # log all the child runs
       for i in range(len(grid_search.cv_results_['params'])):
           with mlflow.start_run(nested=True) as child:
               mlflow.log_params(grid_search.cv_results_["params"][i])
               mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])
```

In the MLflow UI, you’ll be able to:
- Visualize all child runs under a parent
![image](https://github.com/user-attachments/assets/19f470f7-c4c9-46ac-923b-731f68053c5a)

- Compare runs using metrics
![image](https://github.com/user-attachments/assets/9794b668-7b81-4e8a-ae93-da460495d819)

- Identify the best run based on your target metric

---

## ✅ Summary

This repository serves as a reference to:
- Track experiments locally or remotely
- Use autologging for faster experiment capture
- Organize tuning experiments using parent-child runs
- Connect to **DagsHub** for collaborative ML workflows


