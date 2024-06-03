import pandas as pd
import numpy as np

class IndexSetting:
    def __init__(self, data):
        self.data = data

    def definir_index(self):
        print("\nVeuillez sélectionner la colonne d'index dans la liste suivante :")
        for i, column in enumerate(self.data.columns):
            print(f"{i}: {column}")

        while True:
            position_index = input("Entrez la position de la colonne d'index (ou tapez 'exit' pour quitter) : ")
            if position_index.lower() == 'exit':
                print("Sortie...")
                return self.data

            try:
                position_index = int(position_index)
                colonne_index = self.data.columns[position_index]
                self.data.set_index(colonne_index, inplace=True)
                print(f"Colonne d'index '{colonne_index}' définie avec succès.")
                return self.data
            except ValueError:
                print("Erreur : Veuillez entrer une position valide (entier).")
            except IndexError:
                print("Erreur : Position de l'index hors plage. Veuillez réessayer.")
