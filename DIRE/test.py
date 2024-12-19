from utils.config import cfg  # isort: split
import csv
import os
import torch
from utils.eval import get_val_cfg, validate
from utils.utils import get_network

cfg = get_val_cfg(cfg, split="test", copy=False)

assert cfg.ckpt_path, "Please specify the path to the model checkpoint"
model_name = os.path.basename(cfg.ckpt_path).replace(".pth", "")
dataset_root = cfg.dataset_root
rows = []
print(f"'{cfg.exp_name}:{model_name}' model testing on...")

results_dir = os.path.join(cfg.root_dir, "data", "results")
os.makedirs(results_dir, exist_ok=True)
csv_file = os.path.join(results_dir, f"{cfg.exp_name}-{model_name}.csv")

for i, dataset in enumerate(cfg.datasets_test):
    cfg.dataset_root = os.path.join(dataset_root, dataset)
    cfg.datasets = [""]
    model = get_network(cfg.arch)
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_results, y_true, y_pred, filenames = validate(model, cfg)

    # Debugging outputs
    print(f"Dataset: {dataset}")
    print("y_true:", y_true[:10])
    print("y_pred:", y_pred[:10])
    print("filenames:", filenames[:10])

    if i == 0:
        header = ["Dataset", "Filename", "y_true", "y_pred"] + list(test_results.keys())
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        for filename, true_label, pred_label in zip(filenames, y_true, y_pred):
            print(f"Writing to CSV: {dataset}, {filename}, {true_label}, {pred_label}")
            row = [dataset, filename, true_label, pred_label] + list(test_results.values())
            writer.writerow(row)

    print(f"Results written to {csv_file}")
