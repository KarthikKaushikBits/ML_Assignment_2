# Bank Marketing Classification Project

## a. Problem statement
The objective is to predict whether a bank customer will subscribe to a term deposit (`y`) using customer, campaign, and macroeconomic attributes.

## b. Dataset description
- Source: UCI Machine Learning Repository — Bank Marketing.
- File used: `bank-additional-full.csv`.
- Original dataset: 41,188 records and 20 input features plus the target.
- Target: `y` (`yes` or `no`).
- Exact duplicate rows identified and removed: 12.
- Missing values reported by pandas: 0.
- The `duration` feature was excluded because it is only known after the call and can cause target leakage in a prospective prediction task.
- After duplicate removal: 41176 records.

## c. GitHub Repository Link
Add your GitHub repository URL here after creating and uploading the project.

## d. Models used and evaluation metrics

The data was split into 80% training and 20% testing using stratification. Categorical columns were one-hot encoded. Numeric columns were median-imputed and standardized. Preprocessing was fitted within each model pipeline using training data only.

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-----------:|-------:|
| Support Vector Machine |     0.8538 | 0.7851 |      0.4047 |   0.6315 |     0.4933 | 0.4265 |
| Logistic Regression    |     0.8303 | 0.8002 |      0.3591 |   0.6455 |     0.4615 | 0.3928 |
| Naive Bayes            |     0.8082 | 0.7761 |      0.3199 |   0.6239 |     0.4229 | 0.3478 |
| KNN                    |     0.8894 | 0.7408 |      0.5161 |   0.2931 |     0.3739 | 0.3336 |
| Decision Tree          |     0.7489 | 0.7101 |      0.2464 |   0.597  |     0.3489 | 0.2592 |

### Observations about model performance
Use the table above and the confusion matrices in the notebook to write your own observations. Discuss the precision/recall trade-off, AUC, F1-score, and MCC. Do not choose a winner based only on accuracy because the target is imbalanced.

### Overall winner
The model with the highest F1-score on this run is **Support Vector Machine**. Review its AUC and MCC as well before finalizing your conclusion.

## Streamlit application
Run locally with:
```bash
streamlit run app.py
```
Upload `test_data.csv` to see the evaluation metrics, confusion matrix, classification report, and downloadable predictions.

## Repository contents
- `app.py`: Streamlit application.
- `requirements.txt`: Python dependencies.
- `README.md`: project description and evaluation results.
- `test_data.csv`: held-out test data with labels.
- `model/`: saved fitted model pipelines.
- `outputs/`: model metrics, observations, and model file list.

## Reproducibility
The notebook uses `random_state=42` for the train/test split and model configurations.
