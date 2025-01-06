import json
import os
from datasets import Dataset, DatasetDict, load_dataset
import sys




def main(train_data, valid_data, max_files, max_files_val):

    # Create a dataset from the loaded data
    dataset = create_dataset_equal(train_data, max_files)
    
    dataset_val = create_dataset_equal(valid_data, max_files_val)

    # Create a DatasetDict with only a 'train' split
    dataset_dict = DatasetDict({
        "train": dataset,
        "validation": dataset_val
    })

    return dataset_dict

def create_dataset_equal(data, max_num):
    
    count_num = max_num;
    if max_num == None:
        count_num = len(data['data_info'])
        
    data_category = {
        '공학': 0,
        '기타': 0,
        '명칭': 0,
        '보건': 0,
        '사회': 0,
        '산업': 0,
        '예체능': 0,
        '인문': 0,
        '자연': 0,
        '종교': 0
    }
    
    middle = []
    questions = []
    answers = []
    
    for item in data['data_info']:
        if data_category[item['data_category']['middle']] < int(count_num / 10):
            middle.append(item['data_category']['middle'])
            questions.append(item["question"])
            answers.append(item["answer"]["contents"])
            data_category[item['data_category']['middle']] += 1
            
    # Create a dictionary for the dataset
    dataset_dict = {
        "middle" : middle,
        "question": questions,
        "answer": answers
    }
    
    print(data_category)

    # Convert the dictionary into a Dataset
    return Dataset.from_dict(dataset_dict)
            
        
        

if __name__ == "__main__":
    # Directory containing JSON files
    json_train = 'DATASET_QA/3.개방데이터/1.데이터/Training/02.라벨링데이터/SFTlabel.json'
    json_valid = 'DATASET_QA/3.개방데이터/1.데이터/Validation/02.라벨링데이터/SFTlabel.json'
    
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
    
    if (int(sys.argv[2]) == 0):
        max_files_val = None
    else:
        max_files_val = int(sys.argv[2])
        
    

    # Generate DatasetDict
    dataset_dict = main(train_data, valid_data, max_files, max_files_val)

    # Optionally, save the dataset
    if (max_files == None):
        filename = 'dataset-QAequal-ALL'
    else:
        filename = 'dataset-QAequal-'+str(max_files)
    dataset_dict.save_to_disk(filename)

    # Print a sample from the dataset
    print(dataset_dict)
