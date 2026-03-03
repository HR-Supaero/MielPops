import torch
from tqdm import tqdm
import csv

def hierarchical_prediction_to_csv(validation_dataset, model1, model2, model3, device, threshold1=0.0, threshold2=0.0):
    """
    Fait l'inference hiérarchique sur trois modèles :
    - model1 : classes fortes
    - model2 : classes moyennes
    - model3 : classes rares
    Écrit les résultats dans prediction_hierarchical.csv
    """

    models = [model1, model2, model3]
    for m in models:
        m.to(device)
        m.eval()

    id_counter = 0
    predictions_array = []

    with torch.no_grad():
        pbar = tqdm(validation_dataset, desc="Hierarchical Inference")
        for images, labels in pbar:

            images = images.to(device)
            batch_size = images.shape[0]

            # on initialise les images à traiter par le modèle 1
            images_to_process = images
            final_preds = torch.zeros(batch_size, dtype=torch.int32).to(device)
            remaining_mask = torch.ones(batch_size, dtype=torch.bool).to(device)

            # Parcours hiérarchique
            for level, model in enumerate(models):
                if remaining_mask.sum() == 0:
                    break  # toutes les images ont été prédictes

                current_images = images_to_process[remaining_mask]
                logits = model(current_images)
                preds = logits.argmax(dim=1)

                # seuil optionnel pour rerouter vers le prochain niveau
                if level == 0:
                    mask_next = (preds == 99) | (logits.max(dim=1).values < threshold1)
                elif level == 1:
                    mask_next = (preds == 99) | (logits.max(dim=1).values < threshold2)
                else:
                    mask_next = torch.zeros_like(preds, dtype=torch.bool)  # dernier niveau, tout est accepté

                # remplir les predictions finales pour celles qui ne vont pas au niveau suivant
                idx_final = remaining_mask.nonzero(as_tuple=True)[0][~mask_next]
                final_preds[idx_final] = preds[~mask_next]

                # mise à jour du masque pour le prochain niveau
                remaining_mask = mask_next

            # enregistrement batch
            for pred, label in zip(final_preds.cpu().numpy(), labels):
                id_counter += 1
                predictions_array.append({
                    "id": id_counter,
                    "pred": int(pred),
                    "true": int(label)
                })

    # sauvegarde CSV
    output_path = "prediction_hierarchical.csv"
    with open(output_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pred", "true"])
        writer.writeheader()
        writer.writerows(predictions_array)

    print(f"Les predictions hiérarchiques ont été sauvegardées dans {output_path}.")