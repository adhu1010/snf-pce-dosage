#!/usr/bin/env python3
"""
Optimal Superplasticizer Dosage Predictor
Based on Marsh Cone Test Data

Usage:
    python predict_dosage.py --wc 0.40 --sp PCE --sf 5
    python predict_dosage.py --wc 0.35 --sp SNF --sf 0
    python predict_dosage.py --wc 0.42 --sp PCA --sf 10
"""

import argparse
import numpy as np
import joblib
import sys

def load_model():
    """Load the trained model and label encoder"""
    try:
        model = joblib.load('optimal_dosage_predictor.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, label_encoder
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Please train the model first.")
        sys.exit(1)

def predict_optimal_dosage(wc_ratio, sp_type, sf_pct, model, label_encoder):
    """
    Predict optimal superplasticizer dosage
    
    Args:
        wc_ratio (float): Water-to-cement ratio (e.g., 0.35, 0.40, 0.45)
        sp_type (str): Superplasticizer type ('PCE' or 'SNF')
        sf_pct (float): Silica Fume percentage (e.g., 0, 5, 10, 15)
        model: Trained ML model
        label_encoder: Fitted label encoder
    
    Returns:
        float: Predicted optimal dosage as percentage
    """
    # Validate inputs
    if wc_ratio < 0.30 or wc_ratio > 0.50:
        print("⚠️  Warning: W/C ratio outside training range (0.35-0.45)")
    
    if sf_pct < 0 or sf_pct > 20:
        print("⚠️  Warning: Silica Fume % outside typical range (0-15%)")
    
    if sp_type not in ['PCE', 'SNF', 'PCA']:
        raise ValueError("Superplasticizer type must be 'PCE', 'SNF', or 'PCA'")
    
    # Encode and predict
    # Feature order must match training: [wc_ratio, sp_type, silica_fume]
    sp_encoded = label_encoder.transform([sp_type])[0]
    X = np.array([[wc_ratio, sp_encoded, sf_pct]])
    prediction = model.predict(X)[0]
    
    return prediction

def main():
    parser = argparse.ArgumentParser(
        description='Predict optimal superplasticizer dosage for cement paste',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_dosage.py --wc 0.40 --sp PCE --sf 5
  python predict_dosage.py --wc 0.35 --sp SNF --sf 10 --verbose
        """
    )
    
    parser.add_argument('--wc', type=float, required=True,
                        help='Water-to-cement ratio (e.g., 0.35, 0.40, 0.45)')
    parser.add_argument('--sp', type=str, required=True, choices=['PCE', 'SNF', 'PCA'],
                        help='Superplasticizer type (PCE, SNF, or PCA)')
    parser.add_argument('--sf', type=float, default=0.0,
                        help='Silica Fume percentage (default: 0)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information')
    
    args = parser.parse_args()
    
    # Load model
    model, label_encoder = load_model()
    
    # Make prediction
    optimal_dosage = predict_optimal_dosage(args.wc, args.sp, args.sf, model, label_encoder)
    
    # Display results
    print("\n" + "=" * 60)
    print("OPTIMAL DOSAGE PREDICTION")
    print("=" * 60)
    print(f"Water/Cement Ratio:     {args.wc}")
    print(f"Superplasticizer Type:  {args.sp}")
    print(f"Silica Fume:            {args.sf}%")
    print(f"Predicted Optimal Dosage: {optimal_dosage:.3f}%")
    print("=" * 60)
    
    if args.verbose:
        print("\nInterpretation:")
        print(f"• Use {optimal_dosage:.3f}% {args.sp} by weight of binder (cement + SF)")
        print(f"• This is the saturation point estimation")
    
    print()

if __name__ == "__main__":
    main()
