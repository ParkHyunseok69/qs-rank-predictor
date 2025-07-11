import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer 
from sklearn.feature_selection import SelectFromModel 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline 
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import joblib


#DATA PREPROCESSING
university_rankings = pd.read_csv("2026 QS World University Rankings.csv")
university_rankings.drop(["Institution Name"], inplace=True, axis=1) 
university_rankings["Previous Rank"] = university_rankings["Previous Rank"].fillna('New') 
university_rankings["Status"] = university_rankings["Status"].fillna('Unknown') 
university_rankings["Size"] = university_rankings["Size"].fillna(university_rankings["Size"].mode()[0]) 
university_rankings["Research"] = university_rankings["Research"].fillna(university_rankings["Research"].mode()[0]) 
cat_cols = ["Country/Territory", "Region", "Size", "Focus", "Status"] 
num_cols = ["Previous Rank", "AR SCORE", "AR RANK", "ER SCORE", "ER RANK", "FSR SCORE", "FSR RANK", "CPF SCORE", "CPF RANK", "IFR SCORE", "IFR RANK", "ISR SCORE","ISR RANK", 
            "ISD SCORE", "ISD RANK", "IRN SCORE", "IRN RANK", "EO SCORE","EO RANK", "SUS SCORE", "SUS RANK", "Overall SCORE"]
num_cols_fill = ["IFR SCORE", "ISR SCORE", "ISD SCORE", "IRN SCORE", "SUS SCORE"] 
for col in num_cols_fill: 
    university_rankings[col] = university_rankings[col].fillna(university_rankings[col].median())

def clean_rank(val):
    if isinstance(val, str):
        val = val.strip().lower()
        if val in ['unranked', '--', '-', 'new', '']:
            return np.nan
        elif '+' in val:
            return float(val.replace('+', ''))
        elif '-' in val and val.count('-') == 1:
            parts = val.split('-')
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return np.nan
    try:
        return float(val)
    except:
        return np.nan
rank_cols = [col for col in university_rankings.columns if 'RANK' or 'Rank' in col]

for col in rank_cols:
    university_rankings[col] = university_rankings[col].apply(clean_rank)
    university_rankings[col] = university_rankings[col].fillna(university_rankings[col].median())



#MODEL
base_models = [
    ('linear', LinearRegression()),
    ('forest', RandomForestRegressor())
]
meta_model = XGBRegressor()
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=False)


#MODEL FOR SELECTING BEST FEATURES
X_raw = university_rankings.drop(["2026 Rank"], axis=1) 
y = university_rankings["2026 Rank"]

raw_preprocessor = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("num", StandardScaler(), num_cols)])

raw_pipe = Pipeline([('preprocessing', raw_preprocessor), ("feature_selection", SelectFromModel(RandomForestRegressor(), threshold='median')), ("model", stacking_model)]) 
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=30, random_state=10) 
raw_pipe.fit(X_train, y_train)

feature_names = raw_pipe.named_steps['preprocessing'].get_feature_names_out() 
mask = raw_pipe.named_steps['feature_selection'].get_support() 
selected_features = feature_names[mask]
for feat in selected_features:
    print(feat, "-")

raw_selected_features = set()
for feat in selected_features:
    if feat.startswith("num__"):
        raw_selected_features.add(feat.split("__")[1])
    elif feat.startswith("cat__"):
        raw_selected_features.add(feat.split("__")[1].split("_")[0])
raw_selected_features = list(raw_selected_features)


#FINAL MODEL USING THE BEST FEATURES
X_final = university_rankings[raw_selected_features]
y = university_rankings["2026 Rank"]

final_cat = [col for col in raw_selected_features if col in cat_cols]
final_num = [col for col in raw_selected_features if col in num_cols]

final_preprocessor = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), final_cat), ("num", StandardScaler(), final_num)])

final_pipe = Pipeline([('preprocessing', final_preprocessor), ('model', stacking_model)])
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=30, random_state=10) 
final_pipe.fit(X_train, y_train)

joblib.dump(final_pipe, 'uni_rank_predictor.pkl') #SAVE MODEL


#RESULTS
def evaluate(final_pipe, X_test, y_test): 
    global prediction 
    prediction = final_pipe.predict(X_test) 
    mse = mean_squared_error(y_test,prediction) 
    mae = mean_absolute_error(y_test, prediction)
    rmse = root_mean_squared_error(y_test, prediction) 
    r2 = r2_score(y_test, prediction) 
    return mse, mae, rmse, r2 
mse, mae, rmse, r2 = evaluate(final_pipe, X_test, y_test) 
print(f'MSE: {mse}')
print(f'MAE: {mae}') 
print(f'RMSE: {rmse}') 
print(f'R2: {r2}')

def visuals(): 
    plt.figure(figsize=(8, 6)) 
    sns.scatterplot(x=y_test, y=prediction) 
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()]) 
    plt.xlabel("Actual Value") 
    plt.ylabel("Predicted Value") 
    plt.title("Actual VS. Predicted") 
    plt.grid(True) 
    plt.show 
    residuals = y_test - prediction 
    plt.figure(figsize=(8, 6)) 
    sns.histplot(residuals, kde=True, bins=20) 
    plt.axvline(0, color='red', linestyle='--') 
    plt.xlabel("Prediction Error (Residual)") 
    plt.title("Residual Distribution") 
    plt.grid(True) 
    plt.show() 
visuals()