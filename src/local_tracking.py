import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

max_depth = 5
n_estimators = 3

# define experiment 
mlflow.set_experiment('MLOps-Tutorial-1')

with mlflow.start_run():
    # Initialize and train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=2)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric('accuracy', acc)

    # Log parameters
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig('Confusion-matrix.png')

    # log artifacts
    mlflow.log_artifact('Confusion-matrix.png')
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({'Author': 'Kah Yin', 'Project': 'Wine Classfication'})

    # log the model
    mlflow.sklearn.log_model(sk_model=rf, name='Random-Forest-Model')
    print(acc)
