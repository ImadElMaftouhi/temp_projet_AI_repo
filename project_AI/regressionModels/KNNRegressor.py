import shap
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import randint as sp_randint
from evaluationModels.evaluation_regressor import RegressionEvaluator 

class Method_KNN_Regressor:
    def __init__(self):
        self.best_knn = None
        self.explainer = None
        self.X_train_summary = None
    
    def train_knn(self, X_train, y_train, n_iter=100, cv=5, random_state=42):
        print("_________________Entraînement du modèle KNN pour la régression_________________")
        print("Veuillez patienter quelques instants...")

        knn = KNeighborsRegressor()
        param_dist = {
            'n_neighbors': sp_randint(1, 25),
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        }

        random_search = RandomizedSearchCV(knn, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=random_state, n_jobs=-1)
        random_search.fit(X_train, y_train)

        self.best_knn = random_search.best_estimator_
        print(f"Le modèle KNN de régression a été entraîné avec les meilleurs hyperparamètres: {random_search.best_params_}.")

        # Préparer l'explainer SHAP après l'entraînement
        self.X_train_summary = shap.kmeans(X_train, 30)
        self.explainer = shap.KernelExplainer(self.best_knn.predict, self.X_train_summary)

        return self

    def predict(self, X_test):
        if self.best_knn is None:
            raise ValueError("Le modèle n'a pas été entraîné. Veuillez appeler la méthode 'train_knn' d'abord.")
        else:
            print("La prédiction avec les données de test...")

        return self.best_knn.predict(X_test)

    def explain(self, X_instance):
        """
        Explique une instance de données individuelle en utilisant SHAP pour la régression.
        """
        # Vérifier si l'explainer a été initialisé.
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call 'train_knn' with training data first.")
        
        # Calculer les valeurs SHAP pour l'instance spécifique.
        shap_values = self.explainer.shap_values(X_instance)
        return shap_values

    def summary_plot(self):
        """
        Affiche un résumé du plot des valeurs SHAP pour le modèle de régression sur l'ensemble d'entraînement.
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call 'train_knn' with training data first.")
        
        shap_values = self.explainer.shap_values(self.X_train_summary.data)  # Accéder aux données sous-jacentes si DenseData est utilisé
        shap.summary_plot(shap_values, self.X_train_summary.data)

    def run_KNN_regressor(self, X_train, y_train, X_test, y_test):
        print("______________Entraînement du modèle KNN pour la régression______________")
        # Entraînement du modèle
        self.train_knn(X_train, y_train)

        # Prédiction sur les données de test
        y_pred = self.predict(X_test)

        print('_________________Evaluation Metrics_________________')
        # Évaluation du modèle
        evaluator = RegressionEvaluator(y_test, y_pred)  # Assurez-vous que cette classe est définie ailleurs.
        evaluator.evaluation_metrics()
        
        print('_________________Explicabilité du Modèle KNN pour la Régression avec SHAP_________________')
        print('Découvrez comment les différentes caractéristiques influencent les prédictions...')
        # Affichage du summary plot SHAP
        self.summary_plot()
