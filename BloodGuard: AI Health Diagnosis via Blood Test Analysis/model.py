import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def load_data(filename):
    df = pd.read_csv(filename)
    vals = df["Disease_label"].unique()
    mapping = {val: i for i, val in enumerate(vals)}
    df["Disease_label"] = df["Disease_label"].map(mapping)
    return df

class PredictionBloodModel:
    def __init__(self,filename):
        self.model = None
        self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test = None,None,None,None, None, None
        self.df = load_data(filename)
        self.label_mapping ={
            0:"healthy",
            1:"ill"
        }
    def data_training(self):
        self.X = self.df.drop(["Disease_label"], axis=1)
        self.y = self.df["Disease_label"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42)
    def build_model(self):
        self.data_training()
        model = XGBClassifier(n_estimators=100, learning_rate=0.2, max_depth=2, eval_metric="logloss")
        model.fit(self.X_train, self.y_train)
        self.model = model

    def predicttion(self, rbc,hgb, hct, mcv, mch, mchc, wbc, plt):
        new_patient = np.array([[rbc,hgb, hct, mcv, mch, mchc, wbc, plt]])
        pred = self.model.predict(new_patient)
        label_prediction = self.label_mapping[pred[0]]
        print(f"According to your blood test result, you are {label_prediction}")
    def accurracy(self):
        prediction_y = self.model.predict(self.X_test)
        accuracy = accuracy_score(prediction_y,self.y_test)
        print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    model = PredictionBloodModel("krew.csv")
    model.build_model()
    model.accurracy()
    model.predicttion(rbc=5.2,hgb=15.0,hct=45.0,mcv=90.0,mch=30.0,mchc=34.0,wbc=6.0, plt=300.0 )
    model.predicttion(rbc=4.3,hgb=12.5,hct=40.0,mcv=85.0,mch=29.0,mchc=33.0,wbc=14.0,plt=400.0)