#Importing the Libraries
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error

MODEL_FILE = "model.pkl"
PIPELINE = "pipeline.pkl"

def build_pipeline (num_attribs, cat_attribs):
        #For numerical attributes.
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy = "median")),
        ("scaler", StandardScaler())
    ])

    #for categorical attributes.
    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown = "ignore"))
    ])

    #Construct the full pipeline.
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    #Let's train the model
    # 1.Loading the data.
    housing = pd.read_csv("housing.csv")

    # 2. Create a Stratified test set.
    housing['income_cat'] = pd.cut(housing['median_income'], bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels = [1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit( n_splits =  1, test_size = 0.2, random_state = 43)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop("income_cat", axis = 1).to_csv("input.csv", index = False)
        housing = housing.loc[train_index].drop('income_cat', axis = 1)

    # 3. Separate features and labels.
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis = 1)

    # 4. Separate the categorical and numerical attributes(features).
    num_attribs = housing_features.drop("ocean_proximity", axis = 1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)
    print(housing_prepared)

    model = RandomForestRegressor(random_state=43)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE)
    print("Model is trained. Congratulations!")

else:
    #Let's do inference.
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE)

    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value"] = predictions

    input_data.to_csv("output.csv", index = False)
    print("Inference is completed and results saved to output.csv. Enjoy!")


