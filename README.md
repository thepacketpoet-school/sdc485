# SDC485L Final Project: Credit Card Approval and Customer Analytics

**Author:** Haley Archer
**Course:** SDC485L - AI/ML and Data Analytics Methods and Modeling Lab
**Dataset:** AER Credit Card Data (1,319 samples, 12 features)

## Project Overview

This project applies the full machine learning workflow to a credit card dataset, with three analytical tasks: classification of card approval, regression for monthly expenditure, and clustering for customer segmentation. The project progresses through five phases: dataset selection and exploration, preprocessing and EDA, initial ML implementation, hyperparameter tuning and regularization, and finally scenario analysis with real-world implications.

## Analytical Tasks

### 1. Classification: Credit Card Approval (Neural Network)
Predicts whether an applicant would be approved for a credit card based on demographic, financial, and credit history features. Uses a feedforward neural network with dropout regularization and early stopping.

### 2. Regression: Monthly Credit Expenditure (XGBoost)
Predicts the average monthly credit card expenditure based on applicant attributes. Tuned via grid search with L2 regularization through `reg_lambda`.

### 3. Clustering: Customer Segmentation (K-means)
Surfaces natural groupings in the customer base using income, expenditure, credit utilization, active accounts, and time at current address. Optimal K determined via elbow method and silhouette score.

## Summary of Results

| Task | Model | Final Metric |
|---|---|---|
| Classification | Neural Network (32-16, ReLU, Dropout 0.3) | ~98% test accuracy |
| Regression | XGBoost (tuned) | R² improved from baseline via grid search + L2 |
| Clustering | K-means (k=2 by silhouette, k=4 for interpretability) | Silhouette ~0.45 at k=2 |

## Scenario Analysis Findings

Three scenarios were tested on the optimized neural network:

1. **Economic Recession** (income -15%, utilization +25%): Large drop in approval rates, consistent with expected tightening during downturns
2. **Younger Demographic** (age -8 years, fewer accounts): Moderate approval drop, raising fairness considerations for thin-credit-file applicants
3. **Higher Income, Lower Debt** (income +20%, utilization -30%): Modest approval increase, suggesting the model uses multiple correlated signals

Sensitivity analysis on income and credit utilization (share) showed expected monotonic behavior for income and a sharp inflection point for utilization around the 30% threshold, matching established credit scoring practice.

## Dependencies

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- tensorflow / keras
- scipy

Install with: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow scipy`

## Repository Contents

- `Credit_card_data.csv` - the AER credit card dataset
- `Archer_W1_Project_Credit_Card_Report.docx` - Project Part 1 (task definition)
- `Archer_W2_Project_EDA_Preprocessing.ipynb` - Project Part 2 (EDA and preprocessing)
- `Archer_W2_Project_EDA_Report.docx` - Part 2 writeup
- `Archer_W3_Project_ML_Implementation.ipynb` - Project Part 3 (initial models)
- `Archer_W3_Project_ML_Report.docx` - Part 3 writeup
- `Archer_W4_Project_Optimization.ipynb` - Project Part 4 (tuning and regularization)
- `Archer_W4_Project_Optimization_Report.docx` - Part 4 writeup
- `Archer_W5_Project_Scenario_Analysis.ipynb` - Project Part 5 (scenario and sensitivity analysis)
- `Archer_W5_Project_Scenario_Report.docx` - Part 5 writeup

## How to Run

1. Clone this repository
2. Install dependencies (see above)
3. Open any of the `.ipynb` files in Jupyter Notebook or JupyterLab
4. Run cells in order from top to bottom
5. Ensure `Credit_card_data.csv` is in the same directory as the notebook

## License

Academic coursework. Not for commercial use.
