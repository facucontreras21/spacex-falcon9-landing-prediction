# SpaceX Falcon 9 Landing Success Prediction

This project analyzes SpaceX Falcon 9 launch data to understand what factors influence the successful landing of the first stage booster.

The workflow includes data cleaning, exploratory analysis, visualization, and machine learning models to predict landing success.

---

# Project Goal

The objective of this project is to predict whether the Falcon 9 booster will land successfully after launch.

Successful booster recovery significantly reduces launch costs and is a key innovation of SpaceX.

---

# Dataset

The dataset contains information about Falcon 9 launches, including:

- Launch date
- Booster version
- Launch site
- Payload mass
- Orbit
- Customer
- Mission outcome
- Landing outcome

Total launches analyzed:

**101 launches**

---

# Project Structure

Ds-capstone-spacex-falcon9-main

data/
Spacex.csv
spacex_cleaned.csv
model_scores.csv

src/
spacex_portfolio_pipeline.py

notebooks/
(original analysis notebooks)

README.md
requirements.txt


---

# Workflow

The project follows a typical data science pipeline:

1. Load data
2. Clean dataset
3. Create target variable
4. Analyze dataset
5. Visualize data
6. Train machine learning models
7. Save results

---

# Exploratory Analysis

Example insights discovered during analysis:

- Overall landing success rate: **~65%**
- Launch site affects landing success
- Payload mass impacts landing probability
- Different Falcon 9 booster generations have different performance

---

# Machine Learning Models

The following models were trained:

| Model | Description |
|------|------|
| Logistic Regression | Linear model estimating landing probability |
| Decision Tree | Rule-based model using branching decisions |
| Random Forest | Ensemble of multiple decision trees |

The models were trained using features such as:

- Payload mass
- Orbit
- Launch site
- Booster generation
- Launch year

---

# Results

Example model performance:

| Model | Accuracy |
|------|------|
| Logistic Regression | ~0.70 |
| Decision Tree | ~0.73 |
| Random Forest | ~0.76 |

The **Random Forest model performed best** on the test dataset.

---

## Visualization

Landing success rate by launch site:

![Landing Success Rate](data/launch_site_success_rate.png)

Includes simple visual analysis such as:

- Landing success rate by launch site
- Payload mass distribution
- Success rate by orbit

---

# How to Run the Project

Install dependencies:

pip install -r requirements.txt

Run the pipeline:

python src/spacex_portfolio_pipeline.py

This will:

- clean the dataset
- run the analysis
- train models
- save results

Outputs are stored in:

data/spacex_cleaned.csv
data/model_scores.csv

---

# Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

# Author

Facundo Contreras