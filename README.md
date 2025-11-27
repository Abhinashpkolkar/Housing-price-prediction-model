# Housing Price Prediction Model

This project demonstrates how to train, persist, and perform inference on a housing price prediction model using scikit-learn and Joblib.

## Features

- Trains a `RandomForestRegressor` on the California housing dataset.
- Uses a preprocessing pipeline for numerical and categorical features.
- Persists the trained model and pipeline for future inference.
- Automatically splits the data into training and test sets using stratified sampling.
- Performs inference on the test set and saves predictions to `output.csv`.

## Files

- `main.py`: Main script for training and inference.
- `housing.csv`: Input dataset (California housing data).
- `model.pkl`, `pipeline.pkl`: Saved model and pipeline (generated after training).
- `input.csv`: Test set (generated after training).
- `output.csv`: Predictions (generated after inference).

## Usage

1. **Training:**
   - Place `housing.csv` in the project directory.
   - Run `main.py`.  
   - The model and pipeline will be saved as `model.pkl` and `pipeline.pkl`.

2. **Inference:**
   - If `model.pkl` and `pipeline.pkl` exist, running `main.py` will perform inference on `input.csv` and save results to `output.csv`.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib

Install dependencies:
```sh
pip install pandas numpy scikit-learn joblib
```

