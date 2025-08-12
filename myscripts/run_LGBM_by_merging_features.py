import argparse, os, glob, pandas as pd
from qlib.contrib.model.gbdt import LGBModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default="~/.qlib/qlib_data/us_data_alpha158")
    p.add_argument("--outdir", default="~/.qlib/qlib_data/output")
    return p.parse_args()

def merge_features(features_dir):
    print("Merging features...")
    ticker_files = glob.glob(os.path.join(features_dir, "features_*.pkl"))
    
    combined_data = {"train": [], "valid": [], "test": []}
    
    for file_path in ticker_files:
        try:
            ticker_data = pd.read_pickle(file_path)
            for segment in combined_data.keys():
                if segment in ticker_data and ticker_data[segment] is not None:
                    combined_data[segment].append(ticker_data[segment])
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
    
    X_test = data["test"].drop(columns=[c for c in data["test"].columns if c.startswith("LABEL")])
    
    # Train using sklearn-style interface
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(num_leaves=64, n_estimators=500, learning_rate=0.03, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    pred = model.predict(X_test)
    
    # Save results
    pred_df = pd.DataFrame(pred, index=X_test.index, columns=["score"])
    pred_df.reset_index().to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    
    # Save model
    import pickle
    pickle.dump(model, open(os.path.join(out_dir, "lgbm_model.pkl"), "wb"))
    
    print(f"Saved predictions.csv with {len(pred)} predictions")
    print(f"Saved lgbm_model.pkl")

def main():
    args = parse_args()
    features_dir = os.path.expanduser(args.features_dir)
    out_dir = os.path.expanduser(args.outdir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Merge features
    merged_data = merge_features(features_dir)
    
    # Save merged data
    merged_file = os.path.join(out_dir, "merged_features.pkl")
    pd.to_pickle(merged_data, merged_file)
    print(f"Saved merged features to {merged_file}")
    
    # Train model
    if "train" in merged_data and "test" in merged_data:
        train_model(merged_data, out_dir)
    else:
        print("Missing train or test data")

if __name__ == "__main__":
    main()
