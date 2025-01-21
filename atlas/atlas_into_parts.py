import pandas as pd
import math

def split_csv_pandas(input_file, output_prefix, n_parts=10):
    df = pd.read_csv(input_file)
    total_rows = len(df)
    chunk_size = math.ceil(total_rows / n_parts)
    
    for i in range(n_parts):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        # If our slice starts beyond the total rows, we're done
        if start_idx >= total_rows:
            break
        
        chunk_df = df.iloc[start_idx:end_idx]
        chunk_df.to_csv(f"{output_prefix}_part{i+1}.csv", index=False)

# Usage:
if __name__ == "__main__":
    split_csv_pandas("/cbica/home/shahroha/projects/AF-DIT/atlas/atlas.csv", "atlas", 20)