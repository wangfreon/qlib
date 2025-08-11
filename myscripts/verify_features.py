import os, glob, pandas as pd

def verify_features(features_dir="~/.qlib/qlib_data/us_data_alpha158"):
    features_dir = os.path.expanduser(features_dir)
    
    # Check if directory exists
    if not os.path.exists(features_dir):
        print(f"Directory not found: {features_dir}")
        return
    
    # Find all feature files
    feature_files = glob.glob(os.path.join(features_dir, "features_*.pkl"))
    print(f"Found {len(feature_files)} feature files")
    
    if not feature_files:
        print("No feature files found!")
        return
    
    # Check a few sample files
    sample_files = feature_files[:3]
    for file_path in sample_files:
        ticker = os.path.basename(file_path).replace("features_", "").replace(".pkl", "")
        try:
            data = pd.read_pickle(file_path)
            segments = list(data.keys())
            total_rows = sum(len(data[seg]) for seg in segments if data[seg] is not None)
            print(f"{ticker}: {segments} - {total_rows} total rows")
            
            # Show feature columns from first segment
            first_seg = segments[0] if segments else None
            if first_seg and data[first_seg] is not None:
                print(f"  Features: {len(data[first_seg].columns)} columns")
                
        except Exception as e:
            print(f"{ticker}: ERROR - {e}")
    
    print(f"\nTotal tickers processed: {len(feature_files)}")

if __name__ == "__main__":
    verify_features()
