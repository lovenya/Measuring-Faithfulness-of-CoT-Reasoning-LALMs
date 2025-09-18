# data_processing/convert_csv_to_jsonl.py

import csv
import json
import argparse
import os

def infer_and_convert_type(value: str):
    """
    Intelligently infers the data type of a string value and converts it.
    Handles booleans, integers, floats, and strings.
    An empty string is treated as None (null in JSON).
    """
    if value == '':
        return None
    
    # Check for boolean
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    
    # Check for integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Check for float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Default to string
    return value

def convert_csv_to_jsonl(input_path: str, output_path: str):
    """
    Reads a CSV file, converts each row to a JSON object with inferred types,
    and writes the result to a JSONL file.
    """
    print(f"--- Starting Conversion ---")
    print(f"  - Input CSV:  {input_path}")
    print(f"  - Output JSONL: {output_path}")

    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        processed_rows = 0
        with open(input_path, mode='r', encoding='utf-8') as infile, \
             open(output_path, mode='w', encoding='utf-8') as outfile:
            
            # Use DictReader to automatically handle the header row
            reader = csv.DictReader(infile)
            
            for row in reader:
                # Create a new dictionary by applying the type conversion to each value
                processed_row = {key: infer_and_convert_type(val) for key, val in row.items()}
                
                # Convert the processed dictionary to a JSON string and write it to the file
                outfile.write(json.dumps(processed_row) + '\n')
                processed_rows += 1
        
        print(f"\n[SUCCESS] Conversion complete.")
        print(f"  - Total rows processed: {processed_rows}")

    except FileNotFoundError:
        print(f"\n[ERROR] Input file not found at: '{input_path}'")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A robust tool to convert CSV files to JSONL format, with automatic type inference for booleans, integers, and floats.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i', '--input', 
        required=True, 
        help="Path to the input CSV file."
    )
    parser.add_argument(
        '-o', '--output', 
        required=True, 
        help="Path for the output JSONL file."
    )
    args = parser.parse_args()
    
    convert_csv_to_jsonl(args.input, args.output)