import os, glob, argparse

def cleanup_ticker_files(output_dir, keep_combined=True):
    """Remove individual ticker feature files after merging"""
    ticker_files = glob.glob(os.path.join(output_dir, "features_*.pkl"))
    
    if not ticker_files:
        print("No ticker files found to clean up")
        return
    
    print(f"Found {len(ticker_files)} ticker files to remove")
    
    for file_path in ticker_files:
        try:
            os.remove(file_path)
            print(f"Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")
    
    if keep_combined:
        combined_file = os.path.join(output_dir, "combined_features.pkl")
        if os.path.exists(combined_file):
            print(f"Kept: combined_features.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory containing ticker files")
    args = parser.parse_args()
    
    cleanup_ticker_files(args.output_dir)
