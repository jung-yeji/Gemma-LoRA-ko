import json
import os
from datasets import Dataset, DatasetDict, load_dataset
import sys


def load_json_files(json_dir, max_files=None):
    """
    Given a directory, load all JSON files and return a list of dictionaries.
    """
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    if max_files is not None:
        json_files = json_files[:max_files]
    
    data = []
    for json_file in json_files:
        with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
            data.append(json.load(f))
    return data

def create_dataset(data, isvalid=0):
    """
    Convert a list of dictionaries into a Hugging Face Dataset.
    """
    # Define the columns you want to include in the dataset

    question = []
    answer = []
    summary = []

    for item in data:
        question.append(item["dataSet"]["question"]["raw"]["text"])
        answer.append(item["dataSet"]["answer"]["raw"]["text"])
        summary.append(item["dataSet"]["answer"]["summary"]["text"])

        

    # Create a dictionary for the dataset
    dataset_dict = {
        "question": question,
        "answer": answer,
        "summary": summary
    }

    # Convert the dictionary into a Dataset
    return Dataset.from_dict(dataset_dict)

def main(json_train_dir, json_valid_dir, max_files=None, max_files_val=None):
    # Load all JSON files
    data = load_json_files(json_train_dir, max_files)
    
    data_val = load_json_files(json_valid_dir, max_files_val)

    # Create a dataset from the loaded data
    dataset = create_dataset(data, 0)
    
    dataset_val = create_dataset(data_val, 1)

    # Create a DatasetDict with only a 'train' split
    dataset_dict = DatasetDict({
        "train": dataset,
        "validation": dataset_val
    })

    return dataset_dict

if __name__ == "__main__":
    # Directory containing JSON files
    json_train_dir = 'DATASET/129.채용면접 인터뷰 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/PS_F_New'
    json_valid_dir = 'DATASET/129.채용면접 인터뷰 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/PS_F_New'

    if (int(sys.argv[1]) == 0):
        max_files = None
    else:
        max_files = int(sys.argv[1])
    
    if (int(sys.argv[2]) == 0):
        max_files_val = None
    else:
        max_files_val = int(sys.argv[2])

    # Generate DatasetDict
    dataset_dict = main(json_train_dir, json_valid_dir, max_files, max_files_val)

    # Optionally, save the dataset
    if (max_files == None):
        filename = 'dataset-SummaryINT-ALL'
    else:
        filename = 'dataset-SummaryINT-'+str(max_files)
    dataset_dict.save_to_disk(filename)

    # Print a sample from the dataset
    print(dataset_dict)
    print(dataset_dict['train'][384])
    print()
    print(dataset_dict['validation'][487])
    
    # print(medium_dict)
    # print(medium_dict_val)

