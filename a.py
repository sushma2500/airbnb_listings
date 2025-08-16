import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import mean_absolute_error,r2_score
import joblib
import mlflow
import mlflow.sklearn 
mlflow.sklearn.autolog()#Enable Mlflow 

df=pd.read_csv("airbnb_listings.csv")
X=pd.drop(columns=["ListiningId"])
y=["PricePerNight"]
categorical=["City","RoomType"]
numeric=["Bedrooms","Bathrooms","GuestsCapacity","HasWifi","HasAC","DistanceFromCityCenter"]

preprocessor=ColumnTransformer(
    transformers=[("Cat",OneHotEncoder(handle_unknown="ignore"),categorical)],
    remainder="passthrough"
)
model=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("regressor",RandomForestRegressor(n_estimators=100,random_state=42))
])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)
#MLFLOW Experiment
with mlflow.start_run():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    mlflow.log_metric("mae",mae)
    mlflow.log_metric("re_score",r2)
    joblib.run(model,"model.pkl")
    mlflow.log_artifact("model.pkl")
    print(f"Model Saved with MAE={mae},R2={r2}")