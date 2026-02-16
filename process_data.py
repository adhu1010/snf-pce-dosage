import pandas as pd
import numpy as np
import os
import re

def extract_silica_fume_pct(sheet_name):
    """Extract Silica Fume percentage from sheet name."""
    sheet_lower = sheet_name.lower()
    
    # Check for direct percentage match (e.g., "5% SF", "10% Silica Fume")
    match = re.search(r'(\d+)%\s*(sf|silica fume)', sheet_lower)
    if match:
        return float(match.group(1))
    
    # Check for OPC (Ordinary Portland Cement) -> 0% SF
    if 'opc' in sheet_lower:
        return 0.0
        
    # Default/Fallback
    print(f"⚠️  Warning: Could not determine Silica Fume % from '{sheet_name}'. Assuming 0%.")
    return 0.0

def process_data():
    print("🔄 Processing data files with Silica Fume integration...")
    
    # Files to process
    files = {
        'PCE': 'PCE.xlsx',
        'SNF': 'SNF VALUES.xlsx'
    }
    
    all_flow_data = []
    saturation_points = []
    
    for sp_type, filename in files.items():
        if not os.path.exists(filename):
            print(f"⚠️ Warning: {filename} not found. Skipping...")
            continue
            
        print(f"📖 Reading {filename}...")
        try:
            # Read all sheets
            xls = pd.ExcelFile(filename)
            print(f"   Found sheets: {xls.sheet_names}")
            
            for sheet_name in xls.sheet_names:
                sf_pct = extract_silica_fume_pct(sheet_name)
                print(f"   Processing sheet '{sheet_name}' (SF: {sf_pct}%)")
                
                df = pd.read_excel(filename, sheet_name=sheet_name)
                
                # Check for dosage column variations
                dosage_col = None
                for col in df.columns:
                    if 'dosage' in col.lower():
                        dosage_col = col
                        break
                
                if not dosage_col:
                    print(f"❌ Error: Could not find 'Dosage' column in sheet '{sheet_name}'")
                    continue
                    
                # Process each W/C column
                wc_cols = [c for c in df.columns if 'w/c' in c.lower() or 'wc' in c.lower()]
                
                for col in wc_cols:
                    # Extract W/C ratio
                    try:
                        wc_str = col.lower().replace('w/c', '').replace('wc', '').replace('ratio', '').strip()
                        wc_ratio = float(wc_str)
                    except ValueError:
                        continue
                    
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
                        # Reverted tolerance to 0.5s as per user request
                        tolerance = 0.5
                        candidates = subset[subset['flow_time'] <= min_flow + tolerance]
                        saturation_point = candidates.iloc[0]
                        
                        optimal_dosage = saturation_point['dosage']
                        sat_flow = saturation_point['flow_time']
                        
                        print(f"   Found saturation for {sp_type} W/C {wc_ratio} ({sf_pct}% SF): {optimal_dosage:.3f}% (Flow: {sat_flow}s)")
                        
                        saturation_points.append({
                            'sp_type': sp_type,
                            'wc_ratio': wc_ratio,
                            'silica_fume': sf_pct,
                            'optimal_dosage': optimal_dosage,
                            'min_flow_time': sat_flow
                        })
                        
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # Save processed data
    if all_flow_data:
        flow_df = pd.DataFrame(all_flow_data)
        flow_df.to_csv('processed_flow_data.csv', index=False)
        print(f"✅ Saved {len(flow_df)} data points to 'processed_flow_data.csv'")
        
    if saturation_points:
        sat_df = pd.DataFrame(saturation_points)
        sat_df.to_csv('saturation_points.csv', index=False)
        print(f"✅ Saved {len(sat_df)} saturation points to 'saturation_points.csv'")
    else:
        print("❌ No saturation points found!")

if __name__ == "__main__":
    process_data()
