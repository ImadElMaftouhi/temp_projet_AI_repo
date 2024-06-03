import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split
import re

class DataPreprocessor:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Les données fournies ne sont pas un DataFrame.")
        self.data = data

    def infer_frequency(self, date_series):
        """
        Déduit la fréquence la plus courante dans une série de dates.
        
        Parameters:
        date_series (pd.Series): Une série de dates.
        
        Returns:
        str: La fréquence déduite.
        """
        deltas = date_series.diff().dropna().value_counts()
        most_common_delta = deltas.idxmax()
        
        # Inférer la fréquence basée sur la différence la plus commune
        if most_common_delta == pd.Timedelta(days=1):
            return 'D'
        elif most_common_delta == pd.Timedelta(weeks=1):
            return 'W'
        elif most_common_delta in [pd.Timedelta(days=28), pd.Timedelta(days=30), pd.Timedelta(days=31)]:
            # Vérifier si la première date est le début du mois
            if date_series.dt.is_month_start.all():
                return 'MS'
            else:
                return 'M'
        elif most_common_delta in [pd.Timedelta(days=90), pd.Timedelta(days=91), pd.Timedelta(days=92), pd.Timedelta(days=93)]:
            # Vérifier si la première date est le début du trimestre
            if date_series.dt.is_quarter_start.all():
                return 'QS'
            else:
                return 'Q'
        elif most_common_delta in [pd.Timedelta(days=365), pd.Timedelta(days=366)]:
            # Vérifier si la première date est le début de l'année
            if date_series.dt.is_year_start.all():
                return 'AS'
            else:
                return 'A'
        else:
            # Si aucune fréquence spécifique n'est trouvée, retourner None
            return None

    def fill_missing_dates(self, df, date_col, fill_value=0, cat_fill_value='Non applicable'):
        """
        Remplit les dates manquantes dans un DataFrame et remplace les lignes manquantes par des valeurs spécifiées.
        
        Parameters:
        df (pd.DataFrame): Le DataFrame avec des dates manquantes.
        date_col (str): Le nom de la colonne contenant les dates.
        fill_value (int, optional): La valeur avec laquelle remplir les colonnes numériques manquantes. Par défaut 0.
        cat_fill_value (str, optional): La valeur avec laquelle remplir les colonnes catégorielles manquantes. Par défaut 'Non applicable'.
        
        Returns:
        pd.DataFrame: Le DataFrame avec les dates manquantes remplies.
        """
        df[date_col] = pd.to_datetime(df[date_col])

        # Déterminer les dates de début et de fin à partir des données existantes
        start_date = df[date_col].min()
        end_date = df[date_col].max()

        # Inférer la fréquence des dates
        freq = self.infer_frequency(df[date_col])

        # Créer une plage de dates complète en utilisant la fréquence déduite
        all_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Créer un DataFrame avec la plage de dates complète
        df_all_dates = pd.DataFrame(all_dates, columns=[date_col])

        # Fusionner avec le DataFrame original pour identifier les dates manquantes
        df_merged = df_all_dates.merge(df, on=date_col, how='left')

        # Identifier les colonnes catégorielles et numériques
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != date_col]
        numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col != date_col]

        # Remplacer les valeurs NaN par les valeurs de remplissage spécifiées
        df_merged.fillna({col: fill_value for col in numerical_cols}, inplace=True)
        df_merged.fillna({col: cat_fill_value for col in categorical_cols}, inplace=True)

        # Conserver les types de données d'origine
        for col in df.columns:
            if col != date_col: 
                df_merged[col] = df_merged[col].astype(df[col].dtype)

        print(f"Les dates manquantes ont été comblées dans la colonne '{date_col}'. Les colonnes catégorielles ont été remplies avec '{cat_fill_value}'.")

        return df_merged
    
    def prepare_time_series_data (self):
        # Demander à l'utilisateur si le problème à résoudre est de nature chronologique
        is_time_series = input("Le problème à résoudre est-il de nature chronologique (oui/non)? : ").lower().strip()
        if is_time_series == 'oui':
            print("Colonnes disponibles:")
            for i, col in enumerate(self.data.columns):
                print(f"{i+1}: {col}")
            index_col = int(input("Veuillez entrer le numéro de la colonne de date à traiter : ")) - 1
            if index_col < 0 or index_col >= len(self.data.columns):
                print("Index hors de portée.")
                return None

            chosen_col = self.data.columns[index_col]

            if pd.api.types.is_datetime64_any_dtype(self.data[chosen_col]):
                # Si la colonne est de type date, appliquer la méthode pour remplir les dates manquantes
                self.data = self.fill_missing_dates(self.data, chosen_col, fill_value=0, cat_fill_value='Non applicable')
            else:
                print(f"La colonne sélectionnée '{chosen_col}' n'est pas de type date.")
                return None
    
    def convert_date_columns(self):
        # Vérifiez chaque colonne pour voir si elle contient des dates
        date_patterns = [
            re.compile(r'^\d{4}-\d{2}-\d{2}$'),  # YYYY-MM-DD
            re.compile(r'^\d{4}/\d{2}/\d{2}$'),  # YYYY/MM/DD
            re.compile(r'^\d{2}/\d{2}/\d{4}$'),  # DD/MM/YYYY
            re.compile(r'^\d{4}-\d{2}$')  # YYYY-MM
        ]

        for col in self.data.columns:
            try:
                if self.data[col].apply(lambda x: any(pat.match(str(x)) for pat in date_patterns)).all():
                    self.data[col] = pd.to_datetime(self.data[col], errors='coerce', format='%Y-%m')
                    print(f"La colonne '{col}' a été convertie en dates.")
            except Exception as e:
                print(f"Erreur lors de la conversion de la colonne '{col}' en dates : {e}")

    def get_categorical_columns(self):
            # Exclure les colonnes de date et les colonnes numériques
            date_columns = [col for col in self.data.columns if self.data[col].dtype == 'datetime64[ns]']
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            categorical_columns = [col for col in self.data.select_dtypes(exclude=[np.number]).columns if col not in date_columns]
            return categorical_columns
    
    def determine_target(self):
            while True:
                print("\nMerci de sélectionner la variable de sortie à partir de cette liste :")
                for column in self.data.columns:
                    print(column)
                target = input("La variable de sortie : ")
                if target in self.data.columns:
                    print(f"La variable de sortie '{target}' existe dans le fichier.")
                    target_type = self.data[target].dtype

                    # Déterminer le type de problème
                    if target_type == 'object' or pd.api.types.is_categorical_dtype(self.data[target]):
                        print(f"'{target}' est une variable catégorielle. Nous avons un problème de classification.")
                        return target
                    elif pd.api.types.is_numeric_dtype(self.data[target]):
                        print(f"'{target}' est une variable continue. Nous avons un problème de régression.")
                        return target
                    else:
                        print(f"Erreur: La variable de sortie '{target}' a un type inconnu. Merci de réessayer.")
                else:
                    print(f"Erreur: La variable de sortie '{target}' n'existe pas dans le fichier. Merci de réessayer.")

    def display_duplicate_rows(self):
        # Utilisation de 'duplicated' pour identifier les duplicatas, en marquant toutes les occurrences à l'exception de la première
        duplicates = self.data.duplicated(keep='first')
        # Comptage des duplicatas
        num_duplicates = duplicates.sum()
        # Affichage du nombre de lignes dupliquées
        if num_duplicates > 0:
            print(f"Nombre de lignes dupliquées: {num_duplicates}")
        else:
            print("Il n'y a pas de lignes dupliquées.")   

    def remove_duplicates(self):
        duplicated_rows = self.data.duplicated().sum()
        duplicated_columns = 0
        
        # Vérifier les doublons de colonnes
        for i, col in enumerate(self.data.columns):
            for j in range(i+1, len(self.data.columns)):
                if self.data[col].equals(self.data.iloc[:, j]):
                    duplicated_columns += 1
                    print(f"Colonnes dupliquées: {col} et {self.data.columns[j]}")

        if duplicated_columns > 0:
            choice = input(
                "Voulez-vous supprimer les doublons de colonnes?\n"
                "1. Ne rien faire\n"
                "2. Supprimer les colonnes dupliquées\n"
                "Veuillez entrer votre choix (1-2) : "
            ).strip()

            if choice == "1":  # Supprimer les colonnes dupliquées

                #self.data.drop(columns=duplicated_columns, inplace=True)
                self.data.drop_duplicates(inplace=True)
                print("_________________Colonnes dupliquées supprimées_________________")

        if duplicated_rows > 0:
            choice = input(
                "Voulez-vous supprimer les doublons de lignes?\n"
                "1. Oui\n"
                "2. Non\n"
                "Veuillez entrer votre choix (1-2) : "
            ).strip()

            if choice == "1":  # Supprimer les lignes dupliquées
                self.data.drop_duplicates(inplace=True)
                print("_________________Lignes dupliquées supprimées_________________")

    def display_missing_values(self):
        total_rows = len(self.data)
        print("Pourcentage de valeurs manquantes par colonne:")
        for column in self.data.columns:
            missing_count = self.data[column].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / total_rows) * 100
                print(f"{column}: {missing_percentage:.2f}%")
            else:
                print(f"{column}: 0.00%")
                
    def choose_handling_method(self, data_type):
        options = {
            "numeric": "\n1. Supprimer les lignes\n2. Remplacer par la moyenne\n3. Remplacer par la médiane\n4. Remplacer par le mode\n5. Remplacer par le minimum\n6. Remplacer par le maximum",
            "categorical": "\n1. Supprimer les lignes\n2. Remplacer par le mode"
        }
        while True:
            choice = input(
                f"Comment voulez-vous gérer les valeurs manquantes pour les données {data_type}?\n" +
                options[data_type] +
                "\nVeuillez entrer votre choix : "
            ).strip()
            valid_choices = ["1", "2", "3", "4", "5", "6"] if data_type == "numeric" else ["1", "2"]
            if choice in valid_choices:
                print("Les valeurs manquantes ont été traitées.")
                return int(choice)
            else:
                print("Choix invalide. Réessayez.")

    def handle_missing_values(self):
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        non_numeric_columns = self.data.select_dtypes(exclude=[np.number]).columns

        # Vérifier s'il y a des valeurs manquantes dans les colonnes numériques
        if self.data[numeric_columns].isnull().sum().sum() > 0:
            print("Traitement des valeurs manquantes pour les données numériques:")
            numeric_choice = self.choose_handling_method("numeric")
            self.apply_missing_value_strategy(numeric_columns, numeric_choice)

        # Vérifier s'il y a des valeurs manquantes dans les colonnes non numériques
        if self.data[non_numeric_columns].isnull().sum().sum() > 0:
            print("Traitement des valeurs manquantes pour les données catégorielles:")
            categorical_choice = self.choose_handling_method("categorical")
            self.apply_missing_value_strategy(non_numeric_columns, categorical_choice)

    def apply_missing_value_strategy(self, columns, choice):
        if choice == 1:
            self.data.dropna(subset=columns, inplace=True)
        elif choice == 2 and self.data[columns].dtypes.iloc[0] == np.number:
            self.data[columns] = self.data[columns].fillna(self.data[columns].mean())
        elif choice == 3 and self.data[columns].dtypes[0] == np.number:
            self.data[columns] = self.data[columns].fillna(self.data[columns].median())
        elif choice == 2 and self.data[columns].dtypes.iloc[0] != np.number:
            self.data[columns] = self.data[columns].fillna(self.data[columns].mode().iloc[0])
        elif choice == 3 and self.data[columns].dtypes.iloc[0] != np.number:
            self.data[columns] = self.data[columns].fillna("Unknown")
        elif choice in [4, 5, 6]:  # Mode, Min, Max uniquement pour numériques
            if choice == 4:
                self.data[columns] = self.data[columns].fillna(self.data[columns].mode().iloc[0])
            elif choice == 5:
                self.data[columns] = self.data[columns].fillna(self.data[columns].min())
            elif choice == 6:
                self.data[columns] = self.data[columns].fillna(self.data[columns].max())

    def normalize_numeric_columns(self):
        while True:
            # Demander à l'utilisateur quel type de normalisation il souhaite appliquer
            print("Options de normalisation des caractéristiques :")
            print("1. Normalisation Min-Max (redimensionne les données entre 0 et 1)")
            print("2. Standardisation Z-score (moyenne = 0, écart-type = 1)")
            choice = input("Entrez votre choix (1 ou 2) : ").strip()
            
            if choice == '1':
                # Normalisation Min-Max
                scaler = MinMaxScaler()
                # Appliquer le scaler à toutes les colonnes numériques
                for col in self.data.select_dtypes(include=[np.number]).columns:
                    self.data[col] = scaler.fit_transform(self.data[col].values.reshape(-1, 1))
                print("Normalisation Min-Max appliquée avec succès.")
                break  # Sortir de la boucle après une action réussie
            elif choice == '2':
                # Standardisation Z-score
                scaler = StandardScaler()
                for col in self.data.select_dtypes(include=[np.number]).columns:
                    self.data[col] = scaler.fit_transform(self.data[col].values.reshape(-1, 1))
                print("Standardisation Z-score appliquée avec succès.")
                break  # Sortir de la boucle après une action réussie
            else:
                print("Choix invalide. Veuillez entrer '1' pour la Normalisation Min-Max ou '2' pour la Standardisation Z-score.")

        return self.data

    def encode_categorical_columns(self):
        # Convertir les colonnes catégorielles en variables indicatrices
        #self.data = pd.get_dummies(self.data, columns=self.get_categorical_columns(), dtype=int)
        while True:
            print("Options d'encodage pour les colonnes catégorielles :")
            #print("1. Encodage One-Hot")
            print("1. Encodage Label (entier)")
            choice = input("Entrez votre choix : ").strip()

            if choice == '1':
                # Encodage Label (entier)
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                for col in self.get_categorical_columns():
                    self.data[col] = self.data[col].fillna('Missing')
                    self.data[col] = encoder.fit_transform(self.data[col])
                print("Encodage Label (entier) appliqué avec succès.")
                break
            else:
                print("Choix invalide.")
                #print("Choix invalide. Veuillez entrer '1' pour l'encodage One-Hot ou '2' pour l'encodage Label.")

        return self.data

    def select_features(self):
        # Assumer que self.target_column contient le nom de la colonne cible
        target_column = self.target_column  # ou utilisez self.get_target() si la colonne est déterminée ailleurs

        # Afficher toutes les colonnes disponibles à l'exception de la colonne cible
        print("Colonnes disponibles pour la sélection des caractéristiques :")
        available_columns = [col for col in self.data.columns if col != target_column]
        for i, col in enumerate(available_columns):
            print(f"{i + 1}. {col}")

        # Demander à l'utilisateur de choisir les colonnes à inclure comme caractéristiques
        selected_indices = input("Entrez les numéros des colonnes à utiliser comme caractéristiques, séparés par des virgules (e.g., 1, 3, 5): ")
        selected_indices = [int(x.strip()) - 1 for x in selected_indices.split(',') if x.strip().isdigit()]

        # Extraire les caractéristiques choisies en vérifiant que les indices sont valides
        features = [available_columns[i] for i in selected_indices if i < len(available_columns)]

        # Créer le DataFrame des caractéristiques
        X = self.data[features]
        #X = X.drop(columns=[target_column])
        return X
    
    def split_data(self, X, y, default_test_size=0.2, random_state=42):
        # Demander à l'utilisateur d'entrer la taille de l'ensemble de test
        while True:
                test_size_input = input(f"Veuillez entrer la taille de l'ensemble de test (0.0 < taille < 1.0) : ").strip()
                if test_size_input == '':
                    test_size = default_test_size
                    break
                else:
                    try:
                        test_size = float(test_size_input)
                        if 0.0 < test_size < 1.0:
                            if test_size > 0.5:
                                # Demander confirmation si la taille de test est supérieure à 0.5
                                confirmation = input("Vous avez choisi un ensemble de test plus grand que l'ensemble d'entraînement. Êtes-vous sûr ? (oui/non) : ").strip().lower()
                                if confirmation == 'oui':
                                    break
                                else:
                                    print("Veuillez choisir une taille de test inférieure à 0.5.")
                                    continue
                            break
                        else:
                            print("Erreur: La taille doit être entre 0.0 et 1.0, exclusive. Veuillez entrer une valeur valide.")
                    except ValueError:
                        print("Entrée invalide. Veuillez entrer un nombre décimal entre 0.0 et 1.0.")

        # Diviser les données en ensembles d'entraînement et de test
        print("Les données ont été divisées en ensembles d'entraînement et de test avec :")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("\tX_train shape:", X_train.shape)
        print("\tX_test shape:", X_test.shape)
        print("\ty_train shape:", y_train.shape)
        print("\ty_test shape:", y_test.shape)

        return X_train, X_test, y_train, y_test
   
    def preprocess(self):

        # Supprimer les caractères spéciaux des noms de colonnes
        self.data.columns = [re.sub(r'[^a-zA-Z0-9]+', '', col) for col in self.data.columns]

        self.display_duplicate_rows()
        self.remove_duplicates()
        self.display_missing_values()
        self.handle_missing_values()
        self.prepare_time_series_data ()
        #self.normalize_numeric_columns()
        #self.encode_categorical_columns()
        #self.convert_date_columns()
        return self.data

    