# Adsorption Modeling Project

This project develops and compares machine learning and MLP models to predict adsorption performance (`J`) based on parameters like `Bm`, `k`, `l`, and `r`.

## Features

* Data preprocessing and normalization
* Multiple ML models: Random Forest, XGBoost
* Neural Network (PyTorch)
* TorchModelWrapper for unified `.predict()` interface
* K-Fold cross-validation with metric logging
* RandomizedSearchCV for hyperparameter tuning
* Evaluation metrics: MSE, RMSE, MAE, MAPE, R², threshold-based accuracy

## Structure

'''
├── Data/                 # Input datasets
├── FF_code/              # Data generation script
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks for experiments
├── scripts/              # Utility scripts (e.g., training, evaluation)
└── README.md             # Project description
'''

## Requirements

* Python 3.9+
* PyTorch, scikit-learn, XGBoost, numpy, pandas, matplotlib

---

# Проект по моделированию адсорбции

Цель проекта — предсказать эффективность адсорбции (`J`) по параметрам `Bm`, `k`, `l`, `r` с помощью моделей машинного обучения и многослойного перцептрона (MLP).

## Возможности

* Предобработка данных и нормализация
* Модели: Random Forest, XGBoost
* Нейросеть на PyTorch
* Обёртка TorchModelWrapper для .predict()
* Кросс-валидация K-Fold
* Поиск гиперпараметров RandomizedSearchCV
* Метрики: MSE, RMSE, MAE, MAPE, R², Accuracy < threshold

## Структура

```
├── Data/                 # Входные данные
├── FF_code/              # Код для генерации данных
├── models/               # Сохранённые модели
├── notebooks/            # Jupyter-ноутбуки для экспериментов
├── scripts/              # Вспомогательные скрипты (обучение, оценка)
└── README.md             # Описание проекта
```

## Зависимости

* Python 3.9+
* PyTorch, scikit-learn, XGBoost, numpy, pandas, matplotlib
