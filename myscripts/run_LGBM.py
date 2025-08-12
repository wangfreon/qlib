import argparse, os, pandas as pd, pickle
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_files", nargs='+', required=True, help="List of .pkl feature files")
    p.add_argument("--outdir", default="lgbm_output")
    return p.parse_args()

def load_features(feature_files):
    print(f"Loading {len(feature_files)} feature files...")
    
    # If single file, load directly
    if len(feature_files) == 1:
        return pd.read_pickle(feature_files[0])
    
    # Multiple files - merge them
    combined_data = {"train": [], "valid": [], "test": []}
    
    for file_path in feature_files:
        try:
            data = pd.read_pickle(file_path)
            if isinstance(data, dict):
                # Individual ticker format
                for segment in combined_data.keys():
                    if segment in data and data[segment] is not None:
                        combined_data[segment].append(data[segment])
            else:
                # Assume it's already merged format
                print(f"Warning: {file_path} not in expected format")
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
    
    final_data = {}
    for segment, data_list in combined_data.items():
        if data_list:
            final_data[segment] = pd.concat(data_list, axis=0)
            print(f"{segment}: {len(final_data[segment])} rows")
    
    return final_data

def train_model(data, out_dir):
    print("Training LGBM...")
    
    # Prepare data
    X_train = data["train"].drop(columns=[c for c in data["train"].columns if c.startswith("LABEL")])
    y_train = data["train"]["LABEL0"] if "LABEL0" in data["train"].columns else data["train"].iloc[:, -1]
    
    X_valid = data["valid"].drop(columns=[c for c in data["valid"].columns if c.startswith("LABEL")])
    y_valid = data["valid"]["LABEL0"] if "LABEL0" in data["valid"].columns else data["valid"].iloc[:, -1]
    
    X_test = data["test"].drop(columns=[c for c in data["test"].columns if c.startswith("LABEL")])
    y_test = data["test"]["LABEL0"] if "LABEL0" in data["test"].columns else data["test"].iloc[:, -1]
    
    # Train
    model = LGBMRegressor(num_leaves=64, n_estimators=500, learning_rate=0.03, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse', callbacks=[])
    
    # Predict and evaluate
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)
    
    # Calculate traditional metrics
    metrics = {
        "train_mse": mean_squared_error(y_train, train_pred),
        "valid_mse": mean_squared_error(y_valid, valid_pred),
        "test_mse": mean_squared_error(y_test, test_pred),
        "train_mae": mean_absolute_error(y_train, train_pred),
        "valid_mae": mean_absolute_error(y_valid, valid_pred),
        "test_mae": mean_absolute_error(y_test, test_pred)
    }
    
    # Calculate directional accuracy
    def directional_accuracy(y_true, y_pred):
        return ((y_true > 0) == (y_pred > 0)).mean()
    
    def up_accuracy(y_true, y_pred):
        up_mask = y_true > 0
        if up_mask.sum() == 0:
            return 0.0
        return ((y_true[up_mask] > 0) == (y_pred[up_mask] > 0)).mean()
    
    def down_accuracy(y_true, y_pred):
        down_mask = y_true < 0
        if down_mask.sum() == 0:
            return 0.0
        return ((y_true[down_mask] < 0) == (y_pred[down_mask] < 0)).mean()
    
    # Add directional metrics
    metrics.update({
        "train_direction_acc": directional_accuracy(y_train, train_pred),
        "valid_direction_acc": directional_accuracy(y_valid, valid_pred),
        "test_direction_acc": directional_accuracy(y_test, test_pred),
        "train_up_acc": up_accuracy(y_train, train_pred),
        "valid_up_acc": up_accuracy(y_valid, valid_pred),
        "test_up_acc": up_accuracy(y_test, test_pred),
        "train_down_acc": down_accuracy(y_train, train_pred),
        "valid_down_acc": down_accuracy(y_valid, valid_pred),
        "test_down_acc": down_accuracy(y_test, test_pred)
    })
    
    # Save results
    pd.DataFrame(test_pred, index=X_test.index, columns=["prediction"]).reset_index().to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False)
    
    pd.Series(metrics).to_csv(os.path.join(out_dir, "metrics.csv"), header=["value"])
    
    pickle.dump(model, open(os.path.join(out_dir, "lgbm_model.pkl"), "wb"))
    
    print("Results:")
    print("=== Traditional Metrics ===")
    for k, v in metrics.items():
        if not k.endswith("_acc"):
            print(f"  {k}: {v:.6f}")
    
    print("\n=== Directional Accuracy ===")
    for k, v in metrics.items():
        if k.endswith("_acc"):
            print(f"  {k}: {v:.1%}")
    
    print(f"\nSaved files:")
    print(f"  - test_predictions.csv")
    print(f"  - metrics.csv")
    print(f"  - lgbm_model.pkl")

def main():
    args = parse_args()
    out_dir = os.path.expanduser(args.outdir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Expand file paths
    feature_files = [os.path.expanduser(f) for f in args.feature_files]
    
    data = load_features(feature_files)
    
    if "train" not in data or "test" not in data:
        print("ERROR: Missing train or test data")
        return
    
    train_model(data, out_dir)

if __name__ == "__main__":
    main()
