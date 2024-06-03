import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RegressionEvaluator:
    """
    Classe pour évaluer les performances des modèles de régression.
    """
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_mae(self):
        """Calcule le Mean Absolute Error (MAE)."""
        return mean_absolute_error(self.y_true, self.y_pred)

    def calculate_mse(self):
        """Calcule le Mean Squared Error (MSE)."""
        return mean_squared_error(self.y_true, self.y_pred)

    def calculate_rmse(self):
        """Calcule le Root Mean Squared Error (RMSE)."""
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))

    def calculate_r2_score(self):
        """Calcule le coefficient de détermination R^2."""
        return r2_score(self.y_true, self.y_pred)

    def evaluation_metrics(self):
        """Affiche toutes les métriques d'évaluation pour la régression."""
        print("_________________Evaluation Metrics for Regression_________________")
        print(f"Mean Absolute Error (MAE): {self.calculate_mae():.4f}")
        print(f"Mean Squared Error (MSE): {self.calculate_mse():.4f}")
        print(f"Root Mean Squared Error (RMSE): {self.calculate_rmse():.4f}")
        print(f"Coefficient of Determination (R^2): {self.calculate_r2_score():.4f}")
