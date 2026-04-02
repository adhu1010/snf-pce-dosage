import pandas as pd
import numpy as np
import os
import re

def extract_silica_fume_pct(sheet_name):
    """Extract Silica Fume percentage from sheet name."""
    sheet_lower = sheet_name.lower()
    
    # Check for direct percentage match (e.g., "0% SF", "5% SF", "10% SF", "15% SF")
    match = re.search(r'(\d+)%\s*(sf|silica)', sheet_lower)
    if match:
        return float(match.group(1))
    
    # Check for OPC (Ordinary Portland Cement) -> 0% SF
    if 'opc' in sheet_lower:
        return 0.0
        
    # Default/Fallback
    print(f"⚠️  Warning: Could not determine Silica Fume % from '{sheet_name}'. Assuming 0%.")
    return 0.0

def find_dosage_and_wc_cols(df):
    """
    Detect the dosage column and W/C ratio columns from a DataFrame.
    
    Handles two formats:
      1. Named 'Dosage' column with W/C columns like '0.35 w/c ratio'
      2. Unnamed first column (dosage) with W/C columns like '0.4 w/c ratio'
    
    Returns: (dosage_col, wc_cols_dict) where wc_cols_dict maps column_name -> wc_ratio_float
    """
    dosage_col = None
    wc_cols = {}
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Check if this is a dosage column
        if 'dosage' in col_lower:
            dosage_col = col
            continue
        
        # Check if this is a W/C ratio column
        wc_match = re.search(r'([\d.]+)\s*w/c', col_lower)
        if wc_match:
            try:
                wc_ratio = float(wc_match.group(1))
                wc_cols[col] = wc_ratio
            except ValueError:
                pass
            continue
        
        # Alternative: column has 'wc' in the name
        if 'wc' in col_lower:
            try:
                wc_str = col_lower.replace('w/c', '').replace('wc', '').replace('ratio', '').strip()
                wc_ratio = float(wc_str)
                wc_cols[col] = wc_ratio
            except ValueError:
                pass
    
    # If no explicit dosage column found, use the first column (marshcone format)
    if dosage_col is None and len(wc_cols) > 0:
        dosage_col = df.columns[0]
        print(f"      Using first column '{dosage_col}' as dosage (marshcone format)")
    
    return dosage_col, wc_cols

def process_data():
    print("🔄 Processing data from Marsh Cone experiment files...")
    
    # Data sources: MARSHCONE.xlsx (SNF) and PCA_MarshCone_Values.xlsx (PCA)
    files = {
        'SNF': 'MARSHCONE.xlsx',
        'PCA': 'PCA_MarshCone_Values.xlsx'
    }
    
    all_flow_data = []
    saturation_points = []
    
    for sp_type, filename in files.items():
        if not os.path.exists(filename):
            print(f"⚠️ Warning: {filename} not found. Skipping {sp_type}...")
            continue
            
        print(f"\n📖 Reading {filename} ({sp_type})...")
        try:
            xls = pd.ExcelFile(filename)
            print(f"   Found sheets: {xls.sheet_names}")
            
            for sheet_name in xls.sheet_names:
                sf_pct = extract_silica_fume_pct(sheet_name)
                print(f"   Processing sheet '{sheet_name}' (SF: {sf_pct}%)")
                
                df = pd.read_excel(filename, sheet_name=sheet_name)
                
                # Auto-detect dosage and W/C columns
                dosage_col, wc_cols = find_dosage_and_wc_cols(df)
                
                if dosage_col is None:
                    print(f"   ❌ Error: Could not find dosage column in sheet '{sheet_name}'")
                    continue
                
                if not wc_cols:
                    print(f"   ❌ Error: Could not find W/C ratio columns in sheet '{sheet_name}'")
                    continue
                
                print(f"      Dosage col: '{dosage_col}', W/C cols: {list(wc_cols.keys())}")
                
                for col, wc_ratio in wc_cols.items():
                    # Get valid data
                    subset = df[[dosage_col, col]].dropna()
                    subset.columns = ['dosage', 'flow_time']
                    subset = subset.sort_values('dosage')
                    
                    # Add to flow data collection
                    for _, row in subset.iterrows():
                        all_flow_data.append({
                            'sp_type': sp_type,
                            'wc_ratio': wc_ratio,
                            'silica_fume': sf_pct,
                            'dosage': row['dosage'],
                            'flow_time': row['flow_time']
                        })
                    
                    # Identify saturation point
                    if not subset.empty:
                        min_flow = subset['flow_time'].min()
                        tolerance = 0.5  # 0.5s tolerance
                        candidates = subset[subset['flow_time'] <= min_flow + tolerance]
                        saturation_point = candidates.iloc[0]
                        
                        optimal_dosage = saturation_point['dosage']
                        sat_flow = saturation_point['flow_time']
                        
                        print(f"      ✓ Saturation for W/C {wc_ratio} ({sf_pct}% SF): {optimal_dosage:.3f}% (Flow: {sat_flow}s)")
                        
                        saturation_points.append({
                            'sp_type': sp_type,
                            'wc_ratio': wc_ratio,
                            'silica_fume': sf_pct,
                            'optimal_dosage': optimal_dosage,
                            'min_flow_time': sat_flow
                        })
                        
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

    # Save processed data
    if all_flow_data:
        flow_df = pd.DataFrame(all_flow_data)
        flow_df.to_csv('processed_flow_data.csv', index=False)
        print(f"\n✅ Saved {len(flow_df)} data points to 'processed_flow_data.csv'")
        
    if saturation_points:
        sat_df = pd.DataFrame(saturation_points)
        sat_df.to_csv('saturation_points.csv', index=False)
        print(f"✅ Saved {len(sat_df)} saturation points to 'saturation_points.csv'")
        print("\nSaturation Points Summary:")
        print(sat_df.to_string(index=False))
    else:
        print("❌ No saturation points found!")

if __name__ == "__main__":
    process_data()
