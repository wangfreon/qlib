import argparse, os, glob, pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Directory containing feature files")
    p.add_argument("--output_file", default="merged_features.pkl")
    return p.parse_args()

def merge_features(input_dir, output_file):
    print("Merging features...")
    ticker_files = glob.glob(os.path.join(input_dir, "features_*.pkl"))
    print(f"Found {len(ticker_files)} feature files")
    
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
    
    pd.to_pickle(final_data, output_file)
    print(f"Saved merged features to {output_file}")

def main():
    args = parse_args()
    input_dir = os.path.expanduser(args.input_dir)
    output_file = os.path.expanduser(args.output_file)
    
    merge_features(input_dir, output_file)

if __name__ == "__main__":
    main()
