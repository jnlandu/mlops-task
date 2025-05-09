# MLOps Introduction - Iris ML Pipeline

## Project Overview

This project demonstrates the basics of setting up an MLOps pipeline using the Iris dataset and logistic regression. It implements model training, evaluation, and (data) versioning as part of a robust machine learning pipeline.

## Setup Instructions

1. Clone the repository.
2. Create and activate a Python environment:
  ```
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
3. Install the necessary libraries:
  ```
  pip install -r requirements.txt
  ```

## How to Use

- Run the training pipeline:
  ```
  python iris_pipeline_mlflow.py
  ```

- Autolog the model with MLflow:
  ```
  python mlflow_autologging.py
  ```
-  Run the tests :
  ```
  python test_iris_pipeline.py
  ```
## Note :
You will need to set up an MLflow tracking server to log the model artifacts. You can do this by running:
```
mlflow ui
```
Moreoevr, you can change your `dvc` remote storage in the `dvc.yaml` file. For example, to use S3, you can set:
```
dvc remote add -d myremote s3://mybucket/path
```
Or change in the `config` file:


## Requirements

See `requirements.txt` for a full list of dependencies.

## Further Reading
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

## License
This project is licensed under the MIT License. See the LICENSE file for details.