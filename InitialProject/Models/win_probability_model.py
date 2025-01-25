from sklearn.linear_model import LogisticRegression
import pandas as pd

class WinProbabilityModel:
    def __init__(self, model_class):
        self.model_class = model_class  
        self.models = {} 

    def train(self, X_train_scaled, y_train, y_train_win):
        for stamp in range(5): 
            stamp_index = y_train == stamp  
            X_train_stamp = X_train_scaled[stamp_index]
            y_train_stamp_win = y_train_win[stamp_index]

            model = self.model_class()  
            model.fit(X_train_stamp, y_train_stamp_win) 

            self.models[stamp] = model  

    def predict(self, X_test_scaled):
        win_probabilities = {stamp: model.predict_proba(X_test_scaled)[:, 1] for stamp, model in self.models.items()}
        return pd.DataFrame(win_probabilities, index=X_test_scaled.index)
