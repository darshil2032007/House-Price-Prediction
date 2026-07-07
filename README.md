# 🏠 House Price Prediction

An end-to-end **Machine Learning** project that predicts house prices using
**Simple Linear Regression** and **Multiple Linear Regression**, complete with
data cleaning, feature engineering, exploratory data analysis, model
evaluation, and a deployable **Streamlit** web application.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Educational%20Use-green)

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Dataset Description](#-dataset-description)
- [Feature Engineering](#-feature-engineering)
- [Feature Selection & Encoding](#-feature-selection--encoding)
- [Models](#-models)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Requirements](#-requirements)
- [How to Run the Notebook](#-how-to-run-the-notebook)
- [How to Run the Streamlit App](#-how-to-run-the-streamlit-app)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## 📖 Project Overview

This project walks through a complete real-world ML workflow: cleaning raw
housing data, engineering meaningful features, exploring the data visually,
training two regression models, and deploying the better-performing one as
an interactive web app.

## ✨ Features

- Full data cleaning and preprocessing pipeline
- Rich exploratory data analysis (EDA) with 20+ visualizations
- Outlier detection and treatment using the IQR method
- Custom feature engineering (`house_age`, `is_renovated`, `living_lot_ratio`, `total_rooms`)
- One-Hot Encoding of categorical variables
- Two trained models — Simple and Multiple Linear Regression — with a direct performance comparison
- Saved, production-ready model artifacts (`.pkl`)
- Interactive **Streamlit** app for live price predictions

---

## 🔄 Machine Learning Workflow

<details>
<summary>Click to expand the full notebook workflow</summary>

1. Project Overview
2. Import Libraries
3. Load Dataset
4. Data Understanding
5. Data Preprocessing
6. Exploratory Data Analysis (EDA)
7. Outlier Detection & Treatment
8. Feature Construction
9. Feature Selection
10. Feature Encoding
11. Train-Test Split
12. Simple Linear Regression
13. Multiple Linear Regression
14. Model Comparison
15. Save Models
16. Conclusion

</details>

### Data Preprocessing

- Removed the unnecessary `date` column
- Removed rows with invalid prices (`price = 0`)
- Removed invalid houses with `0` bedrooms and `0` bathrooms
- Checked for missing values
- Checked for duplicate values
- Detected outliers using the **IQR method**
- Retained genuine outliers after analysis (they represented real high-value properties, not data errors)

---

## 📊 Dataset Description

The dataset represents residential house sales, with columns describing the
property's physical characteristics, condition, and sale price (the target
variable).

| Category | Examples |
|---|---|
| Target | `price` |
| Structural | `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement`, `floors` |
| Quality | `condition`, `view`, `waterfront` |
| Location | `city`, `statezip` |

---

## 🧩 Feature Engineering

New features were constructed to capture information not directly present in the raw data:

| Feature | Description |
|---|---|
| `house_age` | Age of the house at the time of sale |
| `is_renovated` | Whether the house has been renovated |
| `living_lot_ratio` | Ratio of living area to total lot size |
| `total_rooms` | Combined count of bedrooms and bathrooms |

## 🎯 Feature Selection & Encoding

**Removed** during feature selection: `street`, `country`, `yr_built`, `yr_renovated`, `luxury_score`.
The remaining, most relevant features were retained for model training.

**Encoding:** categorical columns were transformed using **One-Hot Encoding**
via `pd.get_dummies(drop_first=True)`.

**Train-Test Split:** 80% training / 20% testing.

---

## 🤖 Models

| Model | Predictors | Target |
|---|---|---|
| Simple Linear Regression | `sqft_living` | `price` |
| Multiple Linear Regression | All selected & engineered features | `price` |

---

## 📈 Model Performance

| Metric | Simple Linear Regression | Multiple Linear Regression |
|---|---|---|
| MAE | 184,618.17 | **119,756.69** |
| MSE | 76,208,855,246.90 | **42,689,594,114.63** |
| RMSE | 276,059.51 | **206,614.60** |
| R² | 0.5073 | **0.7240** |

**Multiple Linear Regression performed significantly better** than Simple
Linear Regression across every metric, since it leverages many relevant
features about each house instead of relying on a single predictor
(`sqft_living`). It is the model deployed in the Streamlit app.

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── data/
│   ├── raw/                                # Original, unprocessed dataset
│   └── processed/
│       ├── housing_processed.csv           # Cleaned & feature-engineered dataset
│       ├── train.csv                       # Training split (80%)
│       └── test.csv                        # Testing split (20%)
│
├── images/                                 # EDA & evaluation plots
│
├── models/
│   ├── simple_linear_regression.pkl        # Saved Simple Linear Regression model
│   └── multiple_linear_regression.pkl      # Saved Multiple Linear Regression model
│
├── notebook/
│   └── House_Price_Prediction.ipynb        # Full ML workflow notebook
│
├── app.py                                  # Streamlit web application
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation (this file)
```

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/darshil2032007/House-Price-Prediction.git
   cd House-Price-Prediction
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
streamlit
jupyter
```

---

## ▶️ How to Run the Notebook

```bash
cd notebook
jupyter notebook House_Price_Prediction.ipynb
```

Run all cells from top to bottom (**Run → Run All Cells**). The notebook will
clean the raw data, generate EDA plots into `images/`, train both models, and
save them to `models/`.

## 🚀 How to Run the Streamlit App

Once the models have been trained and saved, run the app from the
**project root directory**:

```bash
streamlit run app.py
```

The app loads `models/multiple_linear_regression.pkl` and opens in your
browser (typically at `http://localhost:8501`), where you can:

1. Enter house details
2. Click **Predict Price**
3. View the estimated selling price instantly

---

## 🖼 Screenshots

<details>
<summary>Exploratory Data Analysis</summary>

| Correlation Heatmap | Price Distribution | Top 10 Cities |
|---|---|---|
| `images/correlation_heatmap.png` | `images/price_histogram.png` | `images/top10_cities.png` |

| Price vs Living Area | Price vs Lot Size | Price vs Bathrooms |
|---|---|---|
| `images/price_vs_sqft_living.png` | `images/price_vs_sqft_lot.png` | `images/price_vs_bathrooms.png` |

| Price vs Bedrooms | Price vs Floors | Price vs View |
|---|---|---|
| `images/price_vs_bedrooms.png` | `images/price_vs_floors.png` | `images/price_vs_view.png` |

| Price vs Waterfront | Price Boxplot | All Feature Boxplots |
|---|---|---|
| `images/price_vs_waterfront.png` | `images/price_boxplot.png` | `images/all_boxplots.png` |

</details>

<details>
<summary>Feature Distributions & Countplots</summary>

| Bedrooms | Bathrooms | Floors | Condition | View | Waterfront |
|---|---|---|---|---|---|
| `images/bedrooms_countplot.png` | `images/bathrooms_countplot.png` | `images/floors_countplot.png` | `images/condition_countplot.png` | `images/view_countplot.png` | `images/waterfront_countplot.png` |

| Living Area | Lot Size | Above-Ground Area | Basement Area |
|---|---|---|---|
| `images/sqft_living_distribution.png` | `images/sqft_lot_distribution.png` | `images/sqft_above_distribution.png` | `images/sqft_basement_distribution.png` |

| Year Built | Year Renovated |
|---|---|
| `images/year_built_distribution.png` | `images/year_renovated_distribution.png` |

</details>

<details>
<summary>Model Results</summary>

| Simple Linear Regression Fit | Multiple Linear Regression Predictions |
|---|---|
| `images/simple_regression_line.png` | `images/multiple_regression_prediction.png` |

</details>

---

## 🔮 Future Improvements

- Add more advanced models (Ridge, Lasso, Random Forest, Gradient Boosting) for comparison
- Perform feature scaling and polynomial feature engineering
- Add cross-validation for more robust performance estimates
- Add automated hyperparameter tuning
- Containerize the Streamlit app with Docker for easier deployment
- Add unit tests for the data cleaning and prediction pipeline

---

## ✍️ Author

**Darshil Savaliya**
AI/ML Undergraduate, L.D. College of Engineering (LDCE), Ahmedabad

Feel free to connect, open issues, or submit pull requests to improve this project!

---

## 📄 License

This project is open-source and available for educational use. Feel free to
fork, modify, and build upon it.