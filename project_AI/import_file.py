import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import zipfile
import io
import pandas as pd
import csv
from PIL import Image

class DataImporter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Masquer la fenêtre principale

    def detect_delimiter(self, filename):
        try:
            with open(filename, 'r') as file:
                content = file.read(1024)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(content, delimiters=[',', '\t', ';', '|', ' '])
                return dialect.delimiter
        except csv.Error:
            delimiters = [',', '\t', ';', '|', ' ']
            for delim in delimiters:
                if delim in content:
                    return delim
            print("Aucun délimiteur standard détecté.")
            delim = simpledialog.askstring("Entrée requise", "Entrez le séparateur utilisé dans le fichier :")
            if delim:
                return delim
            else:
                print("Aucun séparateur fourni. Impossible de continuer.")
                return None

    def importer_fichier(self, delim=None):
        print("_________________Chargement des données_________________")
        chemin_fichier = filedialog.askopenfilename(
            filetypes=[("Fichiers CSV", "*.csv"), ("Fichiers Excel", "*.xlsx"), ("Fichiers texte", "*.txt"), ("Fichiers ZIP", "*.zip")])

        if not chemin_fichier:
            print("Aucun fichier sélectionné")
            return None

        extension_fichier = os.path.splitext(chemin_fichier)[1].lower()

        if extension_fichier == '.csv':
            if not delim:
                delim = self.detect_delimiter(chemin_fichier)
            return pd.read_csv(chemin_fichier, delimiter=delim)

        elif extension_fichier == '.txt':
            with open(chemin_fichier, 'r') as file:
                data = file.read()
            return data

        elif extension_fichier == '.xlsx':
            return pd.read_excel(chemin_fichier)

        elif extension_fichier == '.zip':
            with zipfile.ZipFile(chemin_fichier, 'r') as zip_ref:
                fichiers_images = [name for name in zip_ref.namelist() if name.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
                if not fichiers_images:
                    print("Aucun fichier image trouvé dans le zip.")
                    return None

                images = []
                for nom_image in fichiers_images:
                    image_data = zip_ref.read(nom_image)
                    image = Image.open
