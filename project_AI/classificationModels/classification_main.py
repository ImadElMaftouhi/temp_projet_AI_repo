
from classificationModels.KNNClassifier import Method_KNN_Classifier
from classificationModels.DecisionTreeClassifier import Method_DecisionTree
from classificationModels.RandomForestClassifier import Method_RandomForest
#from XAIModels.XAI_KNN import SHAPKNNExplainer


class ClassificationModels_main:
    def __init__(self):
        self.models = {
            1: ('KNN', Method_KNN_Classifier()),
            2: ('Decision Tree', Method_DecisionTree()),
            3: ('Random Forest', Method_RandomForest())
        }

    def select_models(self):
        print("Puisque vous avez un problème de classification")
        print("Merci de sélectionner le(s) modèle(s) que vous voulez utiliser, séparés par des virgules (e.g., 1, 3, 5):")
        for idx, (name, _) in self.models.items():
            print(f"{idx}. {name}")
        selected_indices = input("Votre choix : ").split(',')
        selected_indices = [int(idx.strip()) for idx in selected_indices if idx.strip().isdigit() and int(idx.strip()) in self.models]
        return selected_indices


    def models_selected_classifier(self, model_indices, X_train, y_train, X_test, y_test):
        for idx in model_indices:
            model_name, model = self.models[idx]
            if model_name == 'KNN':
                model.run_KNN_classifier(X_train, y_train, X_test, y_test)
            elif model_name == 'Decision Tree':
                print("...")
            elif model_name == 'Random Forest':
                 print("...")

    def run_models_selected_classifier(self, X_train, y_train, X_test, y_test):
        selected_indices = self.select_models()
        self.models_selected_classifier(selected_indices, X_train, y_train, X_test, y_test)

