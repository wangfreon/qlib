import argparse, os, time, pandas as pd
import qlib
from qlib.config import REG_US
from qlib.data import D
from qlib.contrib.data.handler import Alpha158

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--qlib_dir", default="~/.qlib/qlib_data/us_data")
    p.add_argument("--universe", default="all")
    p.add_argument("--train_start", default="2000-01-01")
    p.add_argument("--train_end", default="2023-12-31")
    p.add_argument("--valid_start", default="2024-01-01")
    p.add_argument("--valid_end", default="2024-12-31")
    p.add_argument("--test_start", default="2025-01-01")
    p.add_argument("--test_end", default="2025-08-01")
    p.add_argument("--outdir", default="~/.qlib/qlib_data/us_data_alpha158")
    return p.parse_args()

def get_ticker_list(universe):
    if universe.strip().lower() == "all":
        return D.list_instruments(D.instruments("all"), as_list=True)
    return [x.strip() for x in universe.split(",") if x.strip()]

def generate_features_for_ticker(ticker, segments, out_dir):
    print(f"Processing {ticker}")
    try:
        handler = Alpha158(
            instruments=[ticker],
            start_time=segments["train"][0],
            end_time=segments["test"][1],
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
        )
        
        features_data = {}
        for segment_name, (start, end) in segments.items():
            data = handler.fetch(selector=(start, end))
            if data is not None and not data.empty:
                features_data[segment_name] = data
        
        if features_data:
            file_path = os.path.join(out_dir, f"features_{ticker}.pkl")
            pd.to_pickle(features_data, file_path)
            
            # Verify file was written
            if os.path.exists(file_path):
                print(f"  Saved: {file_path}")
                return True
            else:
                print(f"  ERROR: File not created for {ticker}")
                return False
        else:
            print(f"  No data for {ticker}")
            return False
    except Exception as e:
        print(f"Failed {ticker}: {e}")
        import traceback
        traceback.print_exc()
    return False

def main():
    args = parse_args()
    out_dir = os.path.expanduser(args.outdir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    qlib.init(provider_uri=os.path.expanduser(args.qlib_dir), region=REG_US)
    
    segments = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "test": (args.test_start, args.test_end),
    }
    
    tickers = get_ticker_list(args.universe)
    print(f"Processing {len(tickers)} tickers")
    
    successful = 0
    for ticker in tickers:
        if generate_features_for_ticker(ticker, segments, out_dir):
            successful += 1
        # Force flush output
        import sys
        sys.stdout.flush()
    
    print(f"Successfully processed {successful}/{len(tickers)} tickers")
    print(f"Files saved in: {out_dir}")

if __name__ == "__main__":
    main()
