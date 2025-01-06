import json
import os
from datasets import Dataset, DatasetDict, load_dataset
import sys


def makeconv(data):
    text = data["title"] + '\n'
    for index in data["text"][0]:
        text += index["sentence"] + "\n"
    summ = data["abstractive"][0]
    
    return (text, summ)

def main(data_store, max_files, max_files_val):

    # Create a dataset from the loaded data
    dataset = create_dataset_equal(data_store, max_files, 0)
    
    dataset_val = create_dataset_equal(data_store, max_files_val, 1)

    # Create a DatasetDict with only a 'train' split
    dataset_dict = DatasetDict({
        "train": dataset,
        "validation": dataset_val
    })

    return dataset_dict

def create_dataset_equal(data_store, max_num, isvalid):
    
    count_num = max_num;
    if max_num == None:
        count_num = len(data_store[isvalid]["documents"])
    
    
    text = []
    summary = []
    

    cycle_num = 0
    for item in data_store[isvalid]["documents"]:
        conv_data = makeconv(item)
        text.append(conv_data[0])
        summary.append(conv_data[1])
        cycle_num += 1
        if cycle_num >= max_num:
            break
        
            
    # Create a dictionary for the dataset
    dataset_dict = {
        "text": text,
        "summary": summary
    }
    

    # Convert the dictionary into a Dataset
    return Dataset.from_dict(dataset_dict)
            
        
        

if __name__ == "__main__":
    # Directory containing JSON files
    json_train = 'DATASET/문서요약 텍스트/Training/train_original.json'
    json_valid = 'DATASET/문서요약 텍스트/Validation/valid_original.json'
    
    
    data_store = []
    

    file_train = open(json_train, 'r', encoding='utf-8')
    file_valid = open(json_valid, 'r', encoding='utf-8')
    data_store.append(json.load(file_train))
    data_store.append(json.load(file_valid))
    file_train.close()
    file_valid.close()


    if (int(sys.argv[1]) == 0):
        max_files = None
    else:
        max_files = int(sys.argv[1])
    
    if (int(sys.argv[2]) == 0):
        max_files_val = None
    else:
        max_files_val = int(sys.argv[2])
        
    

    # Generate DatasetDict
    dataset_dict = main(data_store, max_files, max_files_val)

    # Optionally, save the dataset
    if (max_files == None):
        filename = 'dataset-SummaryLAW-ALL'
    else:
        filename = 'dataset-SummaryLAW-'+str(max_files)
    dataset_dict.save_to_disk(filename)

    # Print a sample from the dataset
    print(dataset_dict)
    print(dataset_dict['train'][287])
    print(dataset_dict['validation'][478])
