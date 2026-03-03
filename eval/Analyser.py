"""
This file contains the Analyser class, which generates our performance report on our evaluation dataset.
data is expected to be a csv in the format:
id, pred, true
"""

import os
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# CLASS_LIST = os.listdir("data/train")
# CLASS_LIST.sort()  # Ensure consistent order
# print(f"Classes: {CLASS_LIST}")

class Analyser():
    def __init__(self, data):
        self.data = data

    def f1_score_avg(self):
        return f1_score(self.data["true"], self.data["pred"], average="macro")
    
    def f1_score_per_class(self):
        return f1_score(self.data["true"], self.data["pred"], average=None)
    
    def generate_report(self):
        report = {
            "f1_score_avg": self.f1_score_avg(),
            "f1_score_per_class": self.f1_score_per_class(),
            "best_f1": np.argmax(self.f1_score_per_class())
        }
        return report
    
def plot_f1_report(report_dict):
    """
    Plot le F1-score par classe + la moyenne.
    """

    f1_scores = report_dict["f1_score_per_class"]
    f1_avg = report_dict["f1_score_avg"]
    best_class = int(report_dict["best_f1"])

    plt.figure(figsize=(12, 5))

    # Barres
    plt.bar(range(len(f1_scores)), f1_scores)

    # Ligne moyenne
    plt.axhline(y=f1_avg)

    # Highlight meilleure classe
    plt.scatter(best_class, f1_scores[best_class])

    plt.title("F1-score par classe")
    plt.xlabel("Classe")
    plt.ylabel("F1-score")
    plt.xticks(range(len(f1_scores)))
    plt.ylim(0, 1)

    plt.show()
    
if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'id': range(100),
        'pred': np.random.randint(0, 50, size=100),
        'true': range(100)
    })
    analyser = Analyser(data)
    report = analyser.generate_report()
    print(report)
    plot_f1_report(report)