import json
import os
from datasets import Dataset, DatasetDict, load_dataset
import sys


def makeconv(data):
    conv = ''
    for dialogue in data["body"]["dialogue"]:
        conv += dialogue["participantID"] + ": " + dialogue["utterance"] + "\n"
    summ = data["body"]["summary"]
    
    return (conv, summ)

def main(data_category, data_store, max_files, max_files_val):

    # Create a dataset from the loaded data
    dataset = create_dataset_equal(data_category, data_store, max_files, 0)
    
    dataset_val = create_dataset_equal(data_category, data_store, max_files_val, 1)

    # Create a DatasetDict with only a 'train' split
    dataset_dict = DatasetDict({
        "train": dataset,
        "validation": dataset_val
    })

    return dataset_dict

def create_dataset_equal(data_category, data_store, max_num, isvalid):
    
    count_num = max_num;
    # if max_num == None:
    #     count_num = len(data['data_info'])
    
    
    conversaton = []
    summary = []
    
    cycle_num = 0
    
    for category in data_category:
        cycle_num = 0
        for item in data_store[category][isvalid]['data']:

            conv_data = makeconv(item)
            conversaton.append(conv_data[0])
            summary.append(conv_data[1])
            cycle_num += 1
            if cycle_num > int(max_num // 9):
                break
    conversaton = conversaton[:max_num]
    summary = summary[:max_num]
            
    # Create a dictionary for the dataset
    dataset_dict = {
        "conversaton": conversaton,
        "summary": summary
    }
    
    print(data_category)

    # Convert the dictionary into a Dataset
    return Dataset.from_dict(dataset_dict)
            
        
        

if __name__ == "__main__":
    # Directory containing JSON files
    json_train = 'DATASET/한국어 대화 요약/Training/[라벨]한국어대화요약_train/'
    json_valid = 'DATASET/한국어 대화 요약/Validation/[라벨]한국어대화요약_valid/'
    
    data_category = [
        '개인및관계',
        '미용과건강',
        '상거래(쇼핑)',
        '시사교육',
        '식음료',
        '여가생활',
        '일과직업',
        '주거와생활',
        '행사'
    ]
    
    data_store = {
        '개인및관계': [],
        '미용과건강': [],
        '상거래(쇼핑)': [],
        '시사교육': [],
        '식음료': [],
        '여가생활': [],
        '일과직업': [],
        '주거와생활': [],
        '행사': []
    }
    
    for category in data_category:
        file_train = open(json_train + category + '.json', 'r', encoding='utf-8')
        file_valid = open(json_valid + category + '.json', 'r', encoding='utf-8')
        data_store[category].append(json.load(file_train))
        data_store[category].append(json.load(file_valid))
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
    dataset_dict = main(data_category, data_store, max_files, max_files_val)

    # Optionally, save the dataset
    if (max_files == None):
        filename = 'dataset-SummaryConv-ALL'
    else:
        filename = 'dataset-SummaryConv-'+str(max_files)
    dataset_dict.save_to_disk(filename)

    # Print a sample from the dataset
    print(dataset_dict)
    print(dataset_dict['train'][287])
    print(dataset_dict['validation'][478])
