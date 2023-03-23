import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from rfpimp import permutation_importances
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data/car_prices.csv')

df['year'] = df['year'].astype('object')
df['price'] = df['price'].astype('float')
df['make'] = df['make'].astype('category')
df['model'] = df['model'].astype('category')
df['year'] = df['year'].astype('category')
df['color'] = df['color'].astype('category')
df['state'] = df['state'].astype('category')

one_hot_encoder = OneHotEncoder()

def automatic_feature_selection(X, y, k):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    baseline_error = accuracy_score(y_val, y_pred)
    feature_importances = model.feature_importances_

    for i in np.argsort(feature_importances)[::-1]:
        # Drop the ith feature
        X_train_new = X_train.drop(X_train.columns[i], axis=1)
        X_val_new = X_val.drop(X_val.columns[i], axis=1)

        # Retrain the model and compute the validation error
        model.fit(X_train_new, y_train)
        y_pred = model.predict(X_val_new)
        new_error = accuracy_score(y_val, y_pred)

        # Check if the new error is better than the baseline error
        if new_error >= baseline_error:
            # If so, update the baseline error and feature importances
            baseline_error = new_error
            feature_importances = model.feature_importances_
        else:
            break
            
    top_k_features = X_train.columns[np.argsort(feature_importances)[::-1]][:k]
    X_train_final = X_train[top_k_features]
    X_val_final = X_val[top_k_features]
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_val_final)
    final_error = accuracy_score(y_val, y_pred)

    return baseline_error, final_error