import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency
from IPython.display import display
import math


class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    def info_data(self):
        print("_________________Visualisation initiale des données_________________")

        print("Cinq premières et derniéres lignes du jeu de données:")
        print(self.data)

        print("\nNombre total de lignes et de colonnes:")
        print(self.data.shape)

        print("\nNoms des colonnes:")
        print(self.data.columns.tolist())

        print("\nTypes de données par colonne:")
        print(self.data.dtypes)

    def ask_change_dtype(self):
            response = input("Voulez-vous changer les types de données des colonnes ? (oui/non): ").lower().strip()
            while response not in ['oui', 'non']:  # S'assure que la réponse est valide
                print("Réponse invalide. Veuillez répondre 'oui' ou 'non'.")
                response = input("Voulez-vous changer les types de données des colonnes ? (oui/non): ").lower().strip()

            if response == 'oui':
                print("Colonnes disponibles:")
                for i, col in enumerate(self.data.columns):
                    print(f"{i+1}: {col}")

                while True:
                    cols_to_change = input("Entrez les numéros des colonnes à modifier, séparés par une virgule (ex: 1,3): ")
                    try:
                        col_indices = [int(x.strip()) - 1 for x in cols_to_change.split(',') if x.strip().isdigit()]
                        if not col_indices or not all(0 <= idx < len(self.data.columns) for idx in col_indices):
                            raise ValueError("Certains indices sont hors de portée ou invalides.")
                        break
                    except ValueError as e:
                        print(f"Erreur: {str(e)}. Veuillez entrer uniquement des numéros de colonnes valides.")

                data_types = {
                    '1': 'int',
                    '2': 'float',
                    '3': 'object',
                    '4': 'bool',
                    '5': 'category',
                    '6': 'datetime64[ns]'
                }

                print("\nTypes de données disponibles:")
                for key, value in data_types.items():
                    print(f"{key}: {value}")

                for index in col_indices:
                    col_name = self.data.columns[index]
                    current_dtype = self.data[col_name].dtype
                    while True:
                        print(f"\nColonne: {col_name}, Type actuel: {current_dtype}")
                        type_choice = input("Entrez le numéro du nouveau type de données (laissez vide pour conserver le type actuel): ").strip()
                        if not type_choice:  # Laisser le type actuel si l'entrée est vide
                            break
                        new_type = data_types.get(type_choice)
                        if new_type:
                            try:
                                self.data[col_name] = self.data[col_name].astype(new_type)
                                print(f"Type de la colonne {col_name} changé en {new_type}.")
                                break
                            except Exception as e:
                                print(f"Erreur lors du changement de type pour la colonne {col_name}: {e}")
                        else:
                            print("Erreur: Choix de type de données invalide.")

            return self.data
  
    def summarize_statistics_and_boxplots(self):
        print("Statistique descriptive pour les variables quantitatives")
        summary = self.data.describe()
        print(summary)

        print("Statistique descriptive pour les variables qualitatives")
        qualitative_summary = self.data.describe(include=['object'])
        print(qualitative_summary)

        continuous_data = self.data.select_dtypes(include=['float64', 'int64'])
        num_cols = 3
        num_rows = math.ceil(len(continuous_data.columns) / num_cols)

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 4*num_rows))
        axes = axes.flatten()
        for i, column in enumerate(continuous_data.columns):
            sns.boxplot(x=continuous_data[column], ax=axes[i])
            axes[i].set_title(f'Boxplot of {column}')
            axes[i].set_xlabel('Value')

        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
    
    def visualize_data(self):
        continuous_columns = self.data.select_dtypes(include=['int64', 'int32', 'float64','float32']).columns
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns

        sns.pairplot(data=self.data[continuous_columns])
        plt.title('Pairplot of Continuous Variables')
        plt.show()

        for column in categorical_columns:
            category_counts = self.data[column].value_counts()
            plt.figure(figsize=(8, 6))
            plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')
            plt.title(column)
            plt.show()

    def relation_categorical(self, target, significance_level=0.05):
        categorical_vars = [col for col in self.data.columns if self.data[col].dtype == 'object']

        if not categorical_vars:
            print("Aucune variable catégorielle trouvée dans le jeu de données.")
            return

        crosstabs = {}
        chi2_stats = {}
        for var in categorical_vars:
            crosstab = pd.crosstab(self.data[var], self.data[target])
            crosstabs[var] = crosstab

            chi2, p, _, _ = chi2_contingency(crosstab)
            chi2_stats[var] = (chi2, p)

        for var, crosstab in crosstabs.items():
            plt.figure(figsize=(8, 6))
            sns.heatmap(crosstab, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f"Relation entre {target} et {var}")
            plt.xlabel(target)
            plt.ylabel(var)
            plt.show()

        ranked_features = sorted(chi2_stats.items(), key=lambda x: x[1][0], reverse=True)
        significant_features = [(var, (chi2, p)) for var, (chi2, p) in ranked_features if p < significance_level]
        print("\nCaractéristiques significatives:")
        for i, (var, (chi2, p)) in enumerate(significant_features):
            print(f"{i + 1}. Variable '{var}' - Chi-deux statistique: {chi2}, P-value: {p}")

        return ranked_features

    def relation_continuous(self, target, significance_level=0.05):
        continuous_vars = [col for col in self.data.columns if self.data[col].dtype != 'object']

        if not continuous_vars:
            print("Aucune variable continue trouvée dans le jeu de données.")
            return

        plt.figure(figsize=(8, 8))
        heatmap = sns.heatmap(self.data[continuous_vars].corr(), annot=True, cmap='Blues', cbar=False)

        t_test_results = {}
        for var in continuous_vars:
            t_stat, p_value = ttest_ind(self.data[self.data[target] == 0][var], self.data[self.data[target] == 1][var])
            t_test_results[var] = (t_stat, p_value)

        ranked_features = sorted(t_test_results.items(), key=lambda x: abs(x[1][0]), reverse=True)

        significant_features = [(var, (t_stat, p)) for var, (t_stat, p) in ranked_features if p < significance_level]
        print("\nCaractéristiques significatives :")
        for i, (var, (t_stat, p)) in enumerate(significant_features):
            print(f"{i + 1}. Variable '{var}' - t-test statistic: {t_stat}, P-value: {p}")

        plt.title("Carte de corrélation des variables continues")
        plt.show()

        return ranked_features

    
    def visualization(self):
        self.info_data()
        self.ask_change_dtype()
        self.summarize_statistics_and_boxplots()
        self.visualize_data()

    def select_features_to_keep(self, target):
        while True:
            # Demander si conserver toutes les variables
            choice = input("Voulez-vous conserver toutes les variables ? 1. Oui 2. Non : ").strip()

            if choice == '1':
                print("Toutes les variables ont été conservées, sauf la variable cible.")
                return self.data.drop(columns=[target])

            elif choice == '2':
                break
            else:
                print("Choix invalide. Veuillez entrer '1' pour conserver toutes les variables ou '2' pour en éliminer.")
        
        # Filtrer les caractéristiques pour exclure la variable cible
        features = [col for col in self.data.columns if col != target]
        print("Caractéristiques disponibles:")
        for idx, feature in enumerate(features):
            print(f"{idx + 1}. {feature}")
        
        selected_indices = input("Entrez les numéros des caractéristiques à supprimer, séparés par une virgule (ex: 1, 3, 5) : ")
        selected_indices = [int(x.strip()) - 1 for x in selected_indices.split(',') if x.strip().isdigit()]

        # Filtrer pour s'assurer que les indices sont valides
        selected_indices = [index for index in selected_indices if index < len(features) and index >= 0]

        features_to_remove = [features[index] for index in selected_indices]
        print("Caractéristiques sélectionnées pour être supprimées : ", features_to_remove)
        self.data = self.data.drop(columns=[target])
        return self.data.drop(columns=features_to_remove)