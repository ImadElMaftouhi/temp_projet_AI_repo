import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score


class ClassifierEvaluator:
    """
    !! Il faut adapte le calcul des métriques de performance en fonction du nombre de classes présentes dans les données.
    Binaire : Utilisée pour les problèmes de classification à deux classes. 
            Les métriques sont basées sur une distinction directe entre les classes positive et négative.

    Macro : Utilisée pour les problèmes de classification multiclasse. 
            Elle calcule les métriques pour chaque classe individuellement et en prend la moyenne, 
            traitant ainsi toutes les classes de manière équitable sans tenir compte de leur fréquence dans 
            les données.
    """
    def __init__(self, y_true, y_pred, y_scores=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores
        self.classification_type = self.determine_classification_type()

    def determine_classification_type(self):
        """Determine if the classification is binary or multiclass based on unique classes."""
        unique_classes = np.unique(self.y_true)
        if len(unique_classes) == 2:
            return 'binary'
        else:
            return 'multiclass'

    def calculate_accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def calculate_precision(self):
        if self.classification_type == 'binary':
            return precision_score(self.y_true, self.y_pred, average='binary')
        else:
            return precision_score(self.y_true, self.y_pred, average='macro')

    def calculate_recall(self):
        if self.classification_type == 'binary':
            return recall_score(self.y_true, self.y_pred, average='binary')
        else:
            return recall_score(self.y_true, self.y_pred, average='macro')

    def calculate_f1_score(self):
        if self.classification_type == 'binary':
            return f1_score(self.y_true, self.y_pred, average='binary')
        else:
            return f1_score(self.y_true, self.y_pred, average='macro')

    def get_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Tracer la matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Matrice de confusion')
        plt.xlabel('Prédit')
        plt.ylabel('Vrai')
        plt.show()
        return cm    

    def evaluation_metrics(self):
        
        print(f"Accuracy: {self.calculate_accuracy():.2%}")
        print(f"Precision: {self.calculate_precision():.2%}")
        print(f"Recall: {self.calculate_recall():.2%}")
        print(f"F1 Score: {self.calculate_f1_score():.2%}")
        print("Confusion Matrix:")
        print(self.get_confusion_matrix())
