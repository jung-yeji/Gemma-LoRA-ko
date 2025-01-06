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

def create_dataset(data):
    """
    Convert a list of dictionaries into a Hugging Face Dataset.
    """
    # Define the columns you want to include in the dataset
    doc_ids = []
    doc_names = []
    authors = []
    publishers = []
    years = []
    passages = []
    passage_cnts = []
    summaries1 = []
    summaries2 = []
    summaries3 = []
    summaries3_cnt = []

    for item in data:
        doc_ids.append(item["Meta(Acqusition)"]["doc_id"])
        doc_names.append(item["Meta(Acqusition)"]["doc_name"])
        authors.append(item["Meta(Acqusition)"]["author"])
        publishers.append(item["Meta(Acqusition)"]["publisher"])
        years.append(item["Meta(Acqusition)"]["publisher_year"])
        passages.append(item["Meta(Refine)"]["passage"])
        passage_cnts.append(item["Meta(Refine)"]["passage_Cnt"])
        summaries1.append(item["Annotation"]["summary1"])
        summaries2.append(item["Annotation"]["summary2"])
        summaries3.append(item["Annotation"]["summary3"])
        summaries3_cnt.append(item["Annotation"]["summary_3_cnt"])

    # Create a dictionary for the dataset
    dataset_dict = {
        "doc_id": doc_ids,
        "doc_name": doc_names,
        "author": authors,
        "publisher": publishers,
        "publisher_year": years,
        "passage": passages,
        "passage_Cnt": passage_cnts,
        "summary1": summaries1,
        "summary2": summaries2,
        "summary3": summaries3,
        "summary_3_cnt": summaries3_cnt
    }

    # Convert the dictionary into a Dataset
    return Dataset.from_dict(dataset_dict)

def main(json_train_dir, json_valid_dir, max_files=None, max_files_val=None):
    # Load all JSON files
    data = load_json_files(json_train_dir, max_files)
    
    data_val = load_json_files(json_valid_dir, max_files_val)

    # Create a dataset from the loaded data
    dataset = create_dataset(data)
    
    dataset_val = create_dataset(data_val)

    # Create a DatasetDict with only a 'train' split
    dataset_dict = DatasetDict({
        "train": dataset,
        "validation": dataset_val
    })

    return dataset_dict

if __name__ == "__main__":
    # Directory containing JSON files
    json_train_dir = 'DATASET/022.요약문 및 레포트 생성 데이터/01.데이터/1.Training/라벨링데이터/01.news_r/20per'
    json_valid_dir = 'DATASET/022.요약문 및 레포트 생성 데이터/01.데이터/2.Validation/라벨링데이터/01.news_r/20per'

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
        filename = 'dataset-ALL'
    else:
        filename = 'dataset-'+str(max_files)
    dataset_dict.save_to_disk(filename)

    # Print a sample from the dataset
    print(dataset_dict)
