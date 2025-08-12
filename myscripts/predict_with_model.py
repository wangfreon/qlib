import pickle
import pandas as pd
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_file", required=True, help="Path to saved model .pkl file")
    p.add_argument("--feature_file", required=True, help="Path to feature .pkl file for prediction")
    p.add_argument("--segment", default="test", help="Segment to predict on (train/valid/test)")
    p.add_argument("--output_file", default="new_predictions.csv")
    return p.parse_args()

def predict_with_saved_model(model_file, feature_file, segment, output_file):
    # Load the saved model
    print(f"Loading model from: {model_file}")
    model = pickle.load(open(model_file, 'rb'))
    
    # Load features
    print(f"Loading features from: {feature_file}")
    data = pd.read_pickle(feature_file)
    
    if segment not in data:
        print(f"Segment '{segment}' not found in data. Available: {list(data.keys())}")
        return
    
    # Prepare features (remove label columns)
    X = data[segment].drop(columns=[c for c in data[segment].columns if c.startswith("LABEL")])
    
    print(f"Making predictions on {len(X)} samples...")
    predictions = model.predict(X)
    
    # Save predictions with index
    pred_df = pd.DataFrame(predictions, index=X.index, columns=["prediction"])
    pred_df.reset_index().to_csv(output_file, index=False)
    
    print(f"Predictions saved to: {output_file}")
    print(f"Sample predictions:")
    print(pred_df.head())

def main():
    args = parse_args()
    predict_with_saved_model(
        args.model_file, 
        args.feature_file, 
        args.segment, 
        args.output_file
    )

if __name__ == "__main__":
    main()
