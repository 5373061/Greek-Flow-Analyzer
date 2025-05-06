import os
import pandas as pd
import subprocess

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("data/options", exist_ok=True)
os.makedirs("data/prices", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Define your tickers - just symbols
tickers = ["AAPL", "MSFT", "QQQ", "SPY", "LULU", "TSLA", "CMG", "WYNN", "ZM", "SPOT"]

# Create a DataFrame and save to CSV
df = pd.DataFrame({"symbol": tickers})
csv_path = "data/my_tickers.csv"
df.to_csv(csv_path, index=False)
print(f"Created CSV file with {len(tickers)} tickers at {csv_path}")

# Run the analysis
cmd = ["python", "run_analysis.py", "batch", csv_path]
print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd)



