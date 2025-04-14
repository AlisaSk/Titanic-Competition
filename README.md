# 🎯 Titanic Survival Prediction (acc 78%)

This project is a solution to the classic [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) challenge on Kaggle. The goal is to predict which passengers survived the Titanic shipwreck using features like age, gender, passenger class, and more.

## 📌 Project Overview

The project is designed as a **modular and production-ready machine learning pipeline**. Each step of the workflow — from data preprocessing to model training and prediction — is separated into clear, reusable functions.

### ✅ Key Steps:
- Data exploration and visualization
- Data preprocessing:
  - Handling missing values
  - Feature engineering
  - Encoding categorical variables
- Building a training pipeline
- Model selection
- Generating predictions and Kaggle submission

## 🧠 Tech Stack

- Python 3.x
- NumPy, Pandas
- Scikit-learn
- Matplotlib & Seaborn

## ⚙️ Production-Ready Code

The code is structured with **modularity and reusability** in mind:
- All major components are wrapped in functions
- Pipeline can be easily extended or transferred to other classification tasks
- Code is clean, organized, and easy to maintain
- Can be easily converted into a package or deployed as a service

## 📈 Result

The final model achieved **78% accuracy** on the public leaderboard. The project serves as a strong baseline for further improvements!

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/main.py
