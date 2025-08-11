# run_pipeline_and_export.py
import argparse, os, time, json, pandas as pd
import qlib
from qlib.config import REG_US
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest_loop
from qlib.contrib.evaluate import risk_analysis

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--qlib_dir", default="~/.qlib/qlib_data/us_data")
    p.add_argument("--universe", default="all", help='Either "all" or comma-separated tickers, e.g. "AAPL,MSFT,AMZN"')
    p.add_argument("--train_start", default="2000-01-01")
    p.add_argument("--train_end",   default="2023-12-31")
    p.add_argument("--valid_start", default="2024-01-01")
    p.add_argument("--valid_end",   default="2024-12-31")
    p.add_argument("--test_start",  default="2025-01-01")
    p.add_argument("--test_end",    default="2025-08-01")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--ndrop", type=int, default=10)
    p.add_argument("--benchmark", default="^GSPC")
    p.add_argument("--outdir", default="~/.qlib/qlib_data/output")
    return p.parse_args()

def ensure_dir(d):
    d = os.path.expanduser(d)  # Expand ~ first
    d = os.path.abspath(d)     # Then get absolute path
    os.makedirs(d, exist_ok=True)
    return d

def parse_universe(u):
    if u.strip().lower() == "all":
        return "all"
    return [x.strip() for x in u.split(",") if x.strip()]

def main():
    args = parse_args()
    out_root = ensure_dir(args.outdir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(os.path.join(out_root, f"run_{stamp}"))

    qlib.init(provider_uri=os.path.expanduser(args.qlib_dir), region=REG_US,
              expression_cache=None, dataset_cache=None)

    uni = parse_universe(args.universe)
    if isinstance(uni, list):
        all_codes = set(D.list_instruments(D.instruments("all"), as_list=True))
        missing = [c for c in uni if c not in all_codes]
        if missing:
            print("[WARN] Not in Qlib package (will be ignored):", missing)
            uni = [c for c in uni if c in all_codes]
        if not uni:
            raise ValueError("No valid tickers in universe after filtering.")

    segments = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "test":  (args.test_start,  args.test_end),
    }

    print("[INFO] Building dataset with Alpha158 features…")
    
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "instruments": uni,
                    "start_time": segments["train"][0],
                    "end_time": segments["test"][1],
                    "fit_start_time": segments["train"][0],
                    "fit_end_time": segments["train"][1],
                }
            },
            "segments": segments
        }
    }
    
    dataset = DatasetH(**dataset_config["kwargs"])
    print(f"[INFO] Dataset created successfully")

    print("[INFO] Training LightGBM…")
    model = LGBModel(
        num_leaves=64,
        n_estimators=500, 
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(dataset)

    try:
        fi = model.get_feature_importance()
        fi.sort_values(ascending=False).to_csv(os.path.join(out_dir, "feature_importance.csv"), header=["importance"])
        print("[INFO] Saved feature_importance.csv")
    except Exception as e:
        print(f"[WARN] Feature importance failed: {e}")

    print("[INFO] Predicting on test…")
    pred = model.predict(dataset, segment="test")
    
    if hasattr(pred, 'reset_index'):
        sig = pred.reset_index()
        if len(sig.columns) == 3:
            sig.columns = ["instrument", "datetime", "score"]
        else:
            sig.columns = ["instrument", "datetime"] + [f"score_{i}" for i in range(len(sig.columns)-2)]
    else:
        sig = pd.DataFrame(pred).reset_index()
        sig.columns = ["instrument", "datetime", "score"]
    
    sig.to_csv(os.path.join(out_dir, "signals_test.csv"), index=False)
    print("[INFO] Saved signals_test.csv")

    print("[INFO] Running analysis…")
    try:
        if hasattr(pred, 'index'):
            pred_series = pred if isinstance(pred, pd.Series) else pred.iloc[:, 0]
        else:
            pred_series = pd.Series(pred)
            
        analysis = {
            "prediction_count": len(pred_series),
            "mean_prediction": float(pred_series.mean()),
            "std_prediction": float(pred_series.std()),
            "min_prediction": float(pred_series.min()),
            "max_prediction": float(pred_series.max())
        }
        
        pd.Series(analysis).to_csv(os.path.join(out_dir, "backtest_analysis.csv"), header=["value"])
        
        with open(os.path.join(out_dir, "backtest_report.json"), "w") as f:
            json.dump({"analysis": analysis}, f, indent=2, default=str)
        print("[INFO] Saved backtest_analysis.csv & backtest_report.json")
        
    except Exception as e:
        print(f"[WARN] Analysis failed: {e}")

    print("\n===== SUMMARY =====")
    print("Output dir:", out_dir)
    print("Key files:")
    print("  - signals_test.csv")
    print("  - feature_importance.csv")
    print("  - backtest_analysis.csv")
    print("  - backtest_report.json")

if __name__ == "__main__":
    main()
