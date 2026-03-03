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

CLASS_LIST = os.listdir("data/train")
CLASS_LIST.sort()  # Ensure consistent order
print(f"Classes: {CLASS_LIST}")

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