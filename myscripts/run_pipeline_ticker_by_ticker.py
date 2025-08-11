import argparse, os, time, json, pandas as pd
import qlib
from qlib.config import REG_US
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--qlib_dir", default="~/.qlib/qlib_data/us_data")
    p.add_argument("--universe", default="all", help='Either "all" or comma-separated tickers')
    p.add_argument("--train_start", default="2000-01-01")
    p.add_argument("--train_end", default="2023-12-31")
    p.add_argument("--valid_start", default="2024-01-01")
    p.add_argument("--valid_end", default="2024-12-31")
    p.add_argument("--test_start", default="2025-01-01")
    p.add_argument("--test_end", default="2025-08-01")
    p.add_argument("--outdir", default="~/.qlib/qlib_data/output")
    return p.parse_args()

def ensure_dir(d):
    d = os.path.expanduser(d)
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d

def get_ticker_list(universe):
    if universe.strip().lower() == "all":
        return D.list_instruments(D.instruments("all"), as_list=True)
    return [x.strip() for x in universe.split(",") if x.strip()]

def generate_features_for_ticker(ticker, segments, out_dir):
    """Generate alpha158 features for a single ticker"""
    print(f"[INFO] Processing ticker: {ticker}")
    
    try:
        handler = Alpha158(
            instruments=[ticker],
            start_time=segments["train"][0],
            end_time=segments["test"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
        )
        
        # Get features for all segments
        features_data = {}
        for segment_name, (start, end) in segments.items():
            try:
                data = handler.fetch(start_time=start, end_time=end)
                if data is not None and not data.empty:
                    features_data[segment_name] = data
                    print(f"  {segment_name}: {len(data)} rows")
                else:
                    print(f"  {segment_name}: No data")
            except Exception as e:
                print(f"  {segment_name}: Error - {e}")
        
        # Save ticker features
        ticker_file = os.path.join(out_dir, f"features_{ticker}.pkl")
        pd.to_pickle(features_data, ticker_file)
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to process {ticker}: {e}")
        return False

def merge_ticker_features(tickers, out_dir):
    """Merge all ticker features into combined datasets"""
    print("[INFO] Merging ticker features...")
    
    combined_data = {"train": [], "valid": [], "test": []}
    
    for ticker in tickers:
        ticker_file = os.path.join(out_dir, f"features_{ticker}.pkl")
        if os.path.exists(ticker_file):
            try:
                ticker_data = pd.read_pickle(ticker_file)
                for segment in combined_data.keys():
                    if segment in ticker_data and ticker_data[segment] is not None:
                        combined_data[segment].append(ticker_data[segment])
            except Exception as e:
                print(f"[WARN] Failed to load {ticker}: {e}")
    
    # Combine and save
    final_data = {}
    for segment, data_list in combined_data.items():
        if data_list:
            final_data[segment] = pd.concat(data_list, axis=0)
            print(f"Combined {segment}: {len(final_data[segment])} rows")
        else:
            print(f"No data for {segment}")
    
    combined_file = os.path.join(out_dir, "combined_features.pkl")
    pd.to_pickle(final_data, combined_file)
    return final_data

def main():
    args = parse_args()
    out_root = ensure_dir(args.outdir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(os.path.join(out_root, f"ticker_run_{stamp}"))
    
    qlib.init(provider_uri=os.path.expanduser(args.qlib_dir), region=REG_US,
              expression_cache=None, dataset_cache=None)
    
    segments = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "test": (args.test_start, args.test_end),
    }
    
    # Get ticker list
    tickers = get_ticker_list(args.universe)
    print(f"[INFO] Processing {len(tickers)} tickers")
    
    # Generate features ticker by ticker
    successful_tickers = []
    for ticker in tickers:
        if generate_features_for_ticker(ticker, segments, out_dir):
            successful_tickers.append(ticker)
    
    print(f"[INFO] Successfully processed {len(successful_tickers)}/{len(tickers)} tickers")
    
    # Merge all features
    combined_data = merge_ticker_features(successful_tickers, out_dir)
    
    if not combined_data or "train" not in combined_data:
        print("[ERROR] No training data available")
        return
    
    # Create dataset from combined features
    print("[INFO] Training model...")
    
    # Create a simple dataset wrapper
    class CombinedDataset:
        def __init__(self, data):
            self.data = data
        
        def prepare(self, segments=None, col_set="feature", data_key="infer"):
            if segments is None:
                segments = ["train", "valid", "test"]
            
            result = {}
            for seg in segments:
                if seg in self.data:
                    df = self.data[seg]
                    if col_set == "feature":
                        # All columns except label
                        feature_cols = [c for c in df.columns if not c.startswith("LABEL")]
                        result[seg] = (df[feature_cols], df.get("LABEL0", pd.Series(index=df.index)))
                    else:
                        result[seg] = df
            return result
    
    dataset = CombinedDataset(combined_data)
    
    # Train model
    model = LGBModel(num_leaves=64, n_estimators=500, learning_rate=0.03)
    
    # Prepare training data
    train_data = dataset.prepare(["train"])
    if "train" in train_data:
        X_train, y_train = train_data["train"]
        model.fit(X_train, y_train)
        
        # Predict on test
        test_data = dataset.prepare(["test"])
        if "test" in test_data:
            X_test, _ = test_data["test"]
            pred = model.predict(X_test)
            
            # Save predictions
            pred_df = pd.DataFrame(pred, index=X_test.index, columns=["score"])
            pred_df.reset_index().to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
            print("[INFO] Saved predictions.csv")
    
    print(f"\n[INFO] Output directory: {out_dir}")
    print("Files created:")
    print("  - features_<ticker>.pkl (individual ticker features)")
    print("  - combined_features.pkl (merged features)")
    print("  - predictions.csv (model predictions)")

if __name__ == "__main__":
    main()
