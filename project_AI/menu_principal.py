import pandas as pd
from import_file import DataImporter
from vis_analyse import DataAnalyzer
from preprocessor import DataPreprocessor

from classificationModels.classification_main import ClassificationModels_main
from regressionModels.regressor_main import RegressorModels_main

class Main:
    def __init__(self):
        self.importer = DataImporter()
        self.mainClassifier = ClassificationModels_main()
        self.mainRegressor = RegressorModels_main()
        self.data_analyzer = None
        self.data = None
    
    def main(self):
       #Chargement des données
        self.data = self.importer.importer_fichier()
        if self.data is None:
            print("Aucune donnée chargée. Sortie... \n")
            return
        print("Données chargées avec succès. \n")

        #Visualisation initiale des données
        data_vis=DataAnalyzer(self.data)
        print(data_vis.visualization())
        
        #Préparation des données        
        if isinstance(self.data, pd.DataFrame):
            # Appeler la fonction pour définir l'index
            preprocessor = DataPreprocessor(self.data)
        
            preprocessed_data = preprocessor.preprocess()
            print("Voici les données prétraitées :")
            print(preprocessed_data)

        # Choix de target et type problème
        target_variable = preprocessor.determine_target()
        target = target_variable
        self.data_analyzer = DataAnalyzer(preprocessed_data)
        print("Exploration des relations entre les caractéristiques et la cible.")
        
        preprocessor.normalize_numeric_columns()
        preprocessor.encode_categorical_columns()

        # Sélectionner les caractéristiques à conserver
        X = self.data_analyzer.select_features_to_keep(target_variable)
        print(X.head(2))
        
        y = preprocessed_data[target_variable]
        print(y)
        
        # Division des données
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        if target != 'numeric' :
            #Appel des modèles de classification
            self.mainClassifier.run_models_selected_classifier(X_train, y_train, X_test, y_test)        

        #else:
            #Appel des modèles de regression
            #self.mainRegressor.run_models_selected_regressor(X_train, y_train, X_test, y_test)        
               
if __name__ == "__main__":
    Main().main()
