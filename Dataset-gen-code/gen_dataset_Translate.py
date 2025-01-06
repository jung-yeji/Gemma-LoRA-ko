import json
import os
from datasets import Dataset, DatasetDict, load_dataset
import sys


def create_dataset(data):
    """
    Convert a list of dictionaries into a Hugging Face Dataset.
    """
    
    sources = []
    destnations = []

    for item in data["data"]:
        sources.append(item["en"])
        destnations.append(item["ko"])

    # Create a dictionary for the dataset
    dataset_dict = {
        "src": sources,
        "dest": destnations
    }

    # Convert the dictionary into a Dataset
    return Dataset.from_dict(dataset_dict)

def main(train_data, valid_data):

    # Create a dataset from the loaded data
    dataset = create_dataset(train_data)
    
    dataset_val = create_dataset(valid_data)

    # Create a DatasetDict with only a 'train' split
    dataset_dict = DatasetDict({
        "train": dataset,
        "validation": dataset_val
    })

    return dataset_dict

if __name__ == "__main__":
    # Directory containing JSON files
    json_train = 'DATASET_Translate/025.일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/일상생활및구어체_영한_train_set.json'
    json_valid = 'DATASET_Translate/025.일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터/01.데이터/2.Validation/라벨링데이터/일상생활및구어체_영한_valid_set.json'
    
    file_train = open(json_train, 'r', encoding='utf-8')
    file_valid = open(json_valid, 'r', encoding='utf-8')
    train_data = json.load(file_train)
    valid_data = json.load(file_valid)
    file_train.close()
    file_valid.close()
    

    if (int(sys.argv[1]) == 0):
        max_files = None
    else:
        max_files = int(sys.argv[1])
        train_data["data"] = train_data["data"][:max_files]
    
    if (int(sys.argv[2]) == 0):
        max_files_val = None
    else:
        max_files_val = int(sys.argv[2])
        valid_data["data"] = valid_data["data"][:max_files_val]
        
    

    # Generate DatasetDict
    dataset_dict = main(train_data, valid_data)

    # Optionally, save the dataset
    if (max_files == None):
        filename = 'dataset-Translate-ALL'
    else:
        filename = 'dataset-Translate-'+str(max_files)
    dataset_dict.save_to_disk(filename)

    # Print a sample from the dataset
    print(dataset_dict)
