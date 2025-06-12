import argparse
import pandas as pd
import requests

def main():
    parser = argparse.ArgumentParser(description="Call prediction API")
    parser.add_argument("--url", required=True, help="Prediction endpoint URL")
    parser.add_argument("--input", required=True, help="CSV file with input records")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if len(df) == 0:
        raise ValueError("Input CSV must contain at least one row")
    data = df.to_dict(orient="records")  # Send all rows
    response = requests.post(args.url, json=data)
    print("Status code:", response.status_code)
    try:
        print("Response:", response.json())
    except ValueError:
        print("Non-JSON response")

if __name__ == "__main__":
    main()
