# Adsorption Modeling Project

This project develops and compares machine learning and MLP models to predict adsorption performance (`J`) based on parameters like `Bm`, `k`, `l`, and `r`.

## Features

* Data preprocessing and normalization
* Multiple ML models: Random Forest, XGBoost
* Neural Network (PyTorch)
* Optuna for hyperparameter tuning (ML models)
* Evaluation metrics: MSE, RMSE, MAE, MAPE, R², threshold-based accuracy

## Structure

```
├── data/                 # Input datasets
├── freefem++/              # Data generation script
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks for experiments
├── src/              # Utility scripts (e.g., training, evaluation)
├── README.md             # Project description
└── setup.py              # Installation script
```

## Requirements

* Python 3.9+
* PyTorch, scikit-learn, XGBoost, numpy, pandas, matplotlib, scipy, typing, joblib

---

# Проект по моделированию адсорбции

Цель проекта — предсказать эффективность адсорбции (`J`) по параметрам `Bm`, `k`, `l`, `r` с помощью моделей машинного обучения и многослойного перцептрона (MLP).

## Возможности

* Предобработка данных и нормализация
* Модели: Random Forest, XGBoost
* Нейросеть на PyTorch
* Поиск гиперпараметров optuna
* Метрики: MSE, RMSE, MAE, MAPE, R², Accuracy < threshold

## Структура

```
├── data/                 # Входные данные
├── freefe++/              # Код для генерации данных
├── models/               # Сохранённые модели
├── notebooks/            # Jupyter-ноутбуки для экспериментов
├── src/              # Вспомогательные скрипты (обучение, оценка)
├── README.md             # Описание проекта
└── setup.py             # Установочный скрипт
```

## Зависимости

* Python 3.9+
* PyTorch, scikit-learn, XGBoost, numpy, pandas, matplotlib, scipy, typing, joblib