import torch
import torchvision
import numpy as np
import csv
from tqdm import tqdm
from model.loader import load_dataset, load_test_loader

def prediction_to_csv(dir,model, img_size=224, batch_size=16) :
    """
    Charge un dataset et retourne le csv des predictions associé.
    """

    # Check for GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU - training will be slow!")

    model = model.to(device)
    model.eval()

    test_loader = load_test_loader(dir, img_size, batch_size)

    # enregistre les predictions
    predictions_array = []

    with torch.no_grad() :
        
        pbar = tqdm(
            test_loader,
            desc="Inference"
        )

        for images, ids in pbar :

            images = images.to(device) # a modifier en fonction du test_set
            img_logits = model(images)


            predictions = img_logits.argmax(dim=1)
            predictions = predictions.cpu().numpy()

            for img_id, pred in zip(ids, predictions):
                predictions_array.append({
                    "id": img_id,
                    "label": int(pred)
                })

        output_path = f"prediction.csv"

        # Sauvegarde CSV
        with open(output_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames = ["id", "label"])
            writer.writeheader()
            writer.writerows(predictions_array)
        print(f"Les predictions ont été sauvegardees dans {output_path}.")