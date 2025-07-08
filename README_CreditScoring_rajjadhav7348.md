
# Credit Scoring Model

## Overview

This project builds a **Credit Scoring Model** that predicts an individual's creditworthiness using financial data such as income, debts, payment history, and credit utilization.

The model uses a **Random Forest Classifier** to classify whether a person is creditworthy (1) or not (0).

---

## Files

- `credit_data.csv`  
  Synthetic dataset containing features and target variable for credit scoring.

- `generate_data.py`  
  Python script to generate the synthetic `credit_data.csv` file.

- `credit_scoring.py`  
  Python script to train the Random Forest model and evaluate its performance.

---

## Requirements

- Python 3.x  
- Packages: pandas, numpy, scikit-learn

Install required packages using pip:

```bash
pip install pandas numpy scikit-learn
```

---

## Usage

1. Generate the dataset (if you don't have your own data):

```bash
python generate_data.py
```

2. Train and evaluate the model:

```bash
python credit_scoring.py
```

---

## Output

The model training script will print:

* First few rows of the dataset
* Classification report (Precision, Recall, F1-score)
* ROC-AUC score (measuring model's ability to discriminate between classes)

---

## Notes

* The dataset is synthetic and generated using simple assumptions. For real-world applications, use authentic financial data and perform thorough feature engineering.
* You can improve the model by tuning hyperparameters or trying other algorithms.

---

## Author

rajjadhav7348

---

Feel free to modify the scripts and dataset to better fit your use case!
