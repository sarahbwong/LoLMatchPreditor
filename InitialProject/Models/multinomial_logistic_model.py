import statsmodels.api as sm

class MultinomialLogisticModel:
    def __init__(self, X_train, y_train):
        self.X_train = sm.add_constant(X_train)
        self.y_train = y_train
        self.model = None

    def train(self, maxiterations):
        self.model = sm.MNLogit(self.y_train, self.X_train)
        self.result = self.model.fit(maxiter=maxiterations)

    def predict(self, X_test):
        X_test = sm.add_constant(X_test)
        return self.result.predict(X_test)
