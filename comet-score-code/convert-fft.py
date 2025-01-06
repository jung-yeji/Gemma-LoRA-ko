import os
import json
import re
import argparse

# Function to extract Rank, file index, translation type, and dataset size from the filename
def extract_file_info(filename):
    # Update regex to match Translate followed by language, dataset size, and index
    trans_type_match = re.search(r'Translate_(\w+)_\d+_\d+\.out$', filename)  # Extract translation type (e.g., cnko, jpko)
    dataset_size_match = re.search(r'_(\d+)_\d+\.out$', filename)  # Match the dataset size before the second '_'
    index_match = re.search(r'_(\d+)\.out$', filename)  # Match the number after the second '_'

    rank = '8'  # Rank is always fixed as 8
    dataset_size = dataset_size_match.group(1) if dataset_size_match else None
    index = index_match.group(1) if index_match else None
    trans_type = trans_type_match.group(1) if trans_type_match else None

    return rank, dataset_size, index, trans_type

# Function to process a single file and extract the relevant data
def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        # Initialize variables for each section
        src, mt_adapter, mt_gemma, ref = "", "", "", ""

        # Extract the necessary information from the lines
        for i, line in enumerate(lines):
            if line.startswith("다음 글을 한국어로 번역해주세요"):
                src = lines[i+2].strip().replace("<end_of_turn>", "").strip()
            elif line.startswith("다음 글을 중국어로 번역해주세요"):
                src = lines[i+2].strip().replace("<end_of_turn>", "").strip()
            elif line.startswith("다음 글을 영어로 번역해주세요"):
                src = lines[i+2].strip().replace("<end_of_turn>", "").strip()
            elif line.startswith("다음 글을 일본어로 번역해주세요"):
                src = lines[i+2].strip().replace("<end_of_turn>", "").strip()
            elif line.startswith("请将以下文章翻译成中文"):
                src = lines[i+2].strip().replace("<end_of_turn>", "").strip()
            elif line.startswith("Adapter Fine-tuned Output"):
                mt_adapter = lines[i+1].strip()
            elif line.startswith("Gemma Output"):
                mt_gemma = lines[i+1].strip()
            elif line.startswith("Label trans result"):
                ref = lines[i+1].strip()

        # Return the extracted data as a dictionary
        return {
            "src": src,           # Input text
            "mt_adapter": mt_adapter,  # Adapter Fine-tuned translation
            "mt_gemma": mt_gemma,  # Gemma translation
            "ref": ref            # Reference translation
        }

# Function to process all files in the folder and convert them to the desired format
def convert_to_format(input_folder):
    rank_data = {}

    for filename in os.listdir(input_folder):
        if filename.startswith("Translate") and filename.endswith(".out"):  # "Translate"로 시작하는 파일만 처리
            filepath = os.path.join(input_folder, filename)

            # Extract the Rank, dataset size, index, and translation type from the filename
            rank, dataset_size, index, trans_type = extract_file_info(filename)
            if rank is not None and dataset_size is not None and index is not None and trans_type is not None:
                # Process the file and get the data
                data = process_file(filepath)
                
                # Store the file index, translation type, and dataset size with the data
                data["file_index"] = index
                data["translation_type"] = trans_type  # Save translation type (e.g., cnko, kocn)
                data["dataset_size"] = dataset_size    # Save dataset size (e.g., 100, 200)

                # Group the data by Rank
                if rank not in rank_data:
                    rank_data[rank] = []  # Initialize list for this rank
                rank_data[rank].append(data)

    return rank_data

# Main function to set up argument parsing
def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process translation result files and convert them into a formatted JSON.")
    
    # Add arguments for input folder and output file
    parser.add_argument('--input_folder', type=str, required=True, help="The folder containing the translation result files.")
    parser.add_argument('--output_file', type=str, required=True, help="The output JSON file to store the formatted data.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Convert all files and save the result in JSON format
    ranked_data = convert_to_format(args.input_folder)
    
    # Save the ranked data to a JSON file
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        json.dump(ranked_data, outfile, ensure_ascii=False, indent=4)
    
    print(f"Ranked data has been successfully formatted and saved to {args.output_file}")

if __name__ == "__main__":
    main()
