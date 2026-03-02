import torch
import torchvision
import numpy as np
import csv

def prediction_to_csv(data_set,model) :

    test_set = load_data_set(data_set)
    predictions_array = []

    for i,img in enumerate(test_set) :
        img_logits = model(img)
        preds = img_logits.argmax(dim=1)
        prediction_entry = {
            "id" : i,
            "label" : preds
        }
        predictions_array.append(prediction_entry)

    prediction_csv = f"result/sumbition_csv/{data_set}_{model}.csv"

    with open(prediction_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = ["id", "label"])
        writer.writeheader()
        writer.writerows(predictions_array)
    print(f"Les predictions ont été sauvegardees dans {fichier_logs}.")

    return