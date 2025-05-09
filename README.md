# MLOps Introduction - Iris ML Pipeline

## Project Overview

This project demonstrates the basics of setting up an MLOps pipeline using the Iris dataset and logistic regression. It implements model training, evaluation, and (data) versioning as part of a robust machine learning pipeline.

## Project Structure

```
├── data/                  # Data directory
├── models/                # Saved model artifacts
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── train.py           # Model training script
│   ├── evaluate.py        # Model evaluation script
│   └── utils.py           # Utility functions
├── tests/                 # Unit tests
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

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
  python src/train.py
  ```

- Evaluate model performance:
  ```
  python src/evaluate.py
  ```

- Experiment with parameters in the notebooks directory

## CI/CD Pipeline

The project includes GitHub Actions workflows for:
- Running tests
- Model validation
- Continuous deployment

## Requirements

See `requirements.txt` for a full list of dependencies.