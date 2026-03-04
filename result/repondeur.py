import torch
from tqdm import tqdm
import csv
import pandas as pd
from model.loader import load_dataset, load_test_loader


# def load_label_mapping(mapping_csv_path):
#     """
#     Retourne :
#     - hier_to_original : dict[int → int]
#     - others_hier_label : int ou None
#     """
#     df = pd.read_csv(mapping_csv_path)

#     hier_to_original = dict(
#         zip(df["hier_label"], df["original_label"])
#     )

#     others_rows = df[df["original_label"] == 99]
#     others_hier_label = (
#         int(others_rows["hier_label"].iloc[0])
#         if len(others_rows) > 0 else None
#     )

#     return hier_to_original, others_hier_label

# def load_label_mapping(mapping_csv_path):
#     df = pd.read_csv(mapping_csv_path)
#     df.columns = df.columns.str.strip()

#     print("Columns found:", df.columns.tolist())

#     # Si l'index a été sauvegardé
#     if "Unnamed: 0" in df.columns:
#         df = df.rename(columns={"Unnamed: 0": "hier_label"})

#     required_cols = {"hier_label", "original_label"}
#     if not required_cols.issubset(df.columns):
#         raise ValueError(
#             f"{mapping_csv_path} must contain {required_cols}, "
#             f"but found {df.columns.tolist()}"
#         )

#     hier_to_original = dict(
#         zip(df["hier_label"], df["original_label"])
#     )

#     return hier_to_original, None

def load_label_mapping(mapping_csv_path):
    df = pd.read_csv(mapping_csv_path)
    df.columns = df.columns.str.strip()

    # 🔥 CAS 1 : hier_label est stocké comme index
    if "hier_label" not in df.columns:
        df = df.reset_index()
        df = df.rename(columns={"index": "hier_label"})

    # 🔥 CAS 2 : index sauvegardé sous "Unnamed: 0"
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "hier_label"})

    required_cols = {"hier_label", "original_label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{mapping_csv_path} must contain {required_cols}, "
            f"but found {df.columns.tolist()}"
        )

    # sécurité : tout en int
    df["hier_label"] = df["hier_label"].astype(int)
    df["original_label"] = df["original_label"].astype(int)

    hier_to_original = dict(
        zip(df["hier_label"], df["original_label"])
    )

    return hier_to_original, None


def hierarchical_prediction_to_csv(
    dir,
    model1,
    model2,
    model3,
    device,
    mapping_lvl1_csv,
    mapping_lvl2_csv,
    mapping_lvl3_csv,
    threshold1=0.0,
    threshold2=0.0,
    output_path="prediction_hierarchical.csv"
):
    """
    Inference hiérarchique à 3 niveaux avec remapping via CSV
    """

    # ===============================
    # Load mappings
    # ===============================
    hier_to_orig_1, others_lvl1 = load_label_mapping(mapping_lvl1_csv)
    hier_to_orig_2, others_lvl2 = load_label_mapping(mapping_lvl2_csv)
    hier_to_orig_3, _ = load_label_mapping(mapping_lvl3_csv)  # no others lvl3

    models = [model1, model2, model3]
    for m in models:
        m.to(device)
        m.eval()

    test_loader = load_test_loader(dir)

    predictions_array = []
    id_counter = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Hierarchical inference")

        for images, labels in pbar:
            images = images.to(device)
            batch_size = images.size(0)

            remaining = torch.ones(batch_size, dtype=torch.bool, device=device)
            final_preds = torch.full(
                (batch_size,), -1, dtype=torch.long, device=device
            )

            # ===============================
            # LEVEL 1
            # ===============================
            if remaining.any():
                imgs = images[remaining]
                logits = model1(imgs)
                conf, preds = logits.max(dim=1)

                to_next = (preds == others_lvl1) | (conf < threshold1)
                idx_global = remaining.nonzero(as_tuple=True)[0]

                for i, p in zip(idx_global[~to_next], preds[~to_next]):
                    final_preds[i] = hier_to_orig_1[int(p)]

                remaining[idx_global] = to_next

            # ===============================
            # LEVEL 2
            # ===============================
            if remaining.any():
                imgs = images[remaining]
                logits = model2(imgs)
                conf, preds = logits.max(dim=1)

                to_next = (preds == others_lvl2) | (conf < threshold2)
                idx_global = remaining.nonzero(as_tuple=True)[0]

                for i, p in zip(idx_global[~to_next], preds[~to_next]):
                    final_preds[i] = hier_to_orig_2[int(p)]

                remaining[idx_global] = to_next

            # ===============================
            # LEVEL 3 (final)
            # ===============================
            if remaining.any():
                imgs = images[remaining]
                preds = model3(imgs).argmax(dim=1)
                idx_global = remaining.nonzero(as_tuple=True)[0]

                for i, p in zip(idx_global, preds):
                    final_preds[i] = hier_to_orig_3[int(p)]

            # ===============================
            # CSV logging
            # ===============================
            for pred, true in zip(final_preds.cpu().numpy(), labels):
                id_counter += 1
                predictions_array.append({
                    "id": id_counter,
                    "pred": int(pred),
                    "true": int(true)
                })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pred", "true"])
        writer.writeheader()
        writer.writerows(predictions_array)

    print(f"CSV hiérarchique sauvegardé : {output_path}")