import joblib
import config
import numpy as np

class HousePriceModel:
    __instance = None

    @staticmethod
    def getInstance():
        if HousePriceModel.__instance == None:
            HousePriceModel()
        return HousePriceModel.__instance

    def __init__(self):
        if HousePriceModel.__instance != None:
            raise Exception("This class is a singleton!")
        self._preprocessor = joblib.load('models/' + config.pipe_to_use)
        self._model = joblib.load('models/' + config.model_to_use)
        HousePriceModel.__instance = self

    def predict(self, X):
        return np.expm1(self._model.predict(self._preprocessor.transform(X))) # np.expm1 needed because during training we log scaled the SalePrice target