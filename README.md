# 🚗 Used Car Price Prediction
Predicts used car prices from listing data (Quikr) using scikit-learn — includes EDA, model comparison, and a deployed Gradio demo.


An end-to-end machine learning project that predicts the resale price of a used car
from real (and messy) listing data scraped from Quikr.com — brand, model, manufacturing
year, kilometers driven, and fuel type.

🔗 **Live demo:** https://car-price-predictiion.onrender.com
📓 **Notebook:** [car_price_prediction.ipynb](./car_price_prediction.ipynb)

## Problem
Given a used car's basic listing details, predict its likely resale price in INR —
a supervised regression task.

## What this project covers
- **Data cleaning** — the raw scrape has junk years, prices stored as text
  (`"Ask For Price"`), mileage with commas and units, missing values, and duplicates.
  Cleaning this messy real-world data was most of the work.
- **EDA** — visualized how price relates to year, mileage, brand, and fuel type.
- **Modeling** — built a `ColumnTransformer` + `Pipeline` with one-hot encoding,
  compared Linear Regression vs Random Forest.
- **Deployment** — saved the trained pipeline and wrapped it in an interactive
  Gradio web app, deployed live on Render.

## Results
| Model                 | R²   | MAE (₹) |
|---------------------- |------|---------|
| **Linear Regression** | 0.59 | 143,175 |
| Random Forest         | 0.56 | 132,426 |

Linear Regression was chosen as the final model for its stronger R².
