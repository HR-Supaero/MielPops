import os

def compute_class_weights(path):
    """
    Compute class weights based on the number of files in each subdirectory.
    Args:
        path (str): The path to the main directory containing class subdirectories.
    Returns:
        torch.Tensor: A tensor of class weights.
    """
    
    # 1. Get and sort subdirectories alphabetically (case-insensitive)
    subdirs = sorted(
        [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))],
        key=str.lower
    )
    
    # 2. Count files in each (Directly in the function)
    counts = {}
    for folder in subdirs:
        sub_path = os.path.join(path, folder)
        file_count = len([f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f))])
        counts[folder] = file_count

    total_files = sum(counts.values())
    num_classes = len(counts)

    # 3. Calculate weights with Zero-Division safety
    # Formula: total_samples / (num_classes * class_samples)
    weights = {}
    for folder, count in counts.items():
        if count > 0:
            weights[folder] = total_files / (num_classes * count)
        else:
            weights[folder] = 0.0  # Assign null weight for empty folders
            
    return weights