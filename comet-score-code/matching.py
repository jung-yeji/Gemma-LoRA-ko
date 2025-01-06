import json
import csv
import os
import fasttext
import argparse

# JSON 데이터를 불러오는 함수
def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# FastText 모델을 로드하는 함수
def load_fasttext_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FastText 모델 파일이 존재하지 않습니다: {model_path}")
    return fasttext.load_model(model_path)

# CSV 파일에서 데이터를 읽고, 업데이트된 데이터를 새로운 파일로 저장하는 함수
def create_updated_csv(input_csv, invalid_combinations, output_csv):
    updated_rows = []
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {input_csv}")
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더를 읽음
        updated_rows.append(header)  # 헤더 추가
        
        for row in reader:
            # Rank, File_Index, Translation_Type, Dataset_Size, Comet_Score_mt_adapter
            rank, file_index, trans_type, dataset_size, score = row[:5]
            combination = (file_index, trans_type, dataset_size)
            
            if combination in invalid_combinations:
                row[4] = '0'  # Score를 0으로 설정
            updated_rows.append(row)
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)  # 모든 데이터 작성
    print(f"새로운 CSV 파일이 생성되었습니다: {output_csv}")

# 언어를 감지하고 유효하지 않은 엔트리를 식별하는 함수
def detect_invalid_combinations(data, lang_model):
    invalid_combinations = set()
    
    for key, entries in data.items():
        for entry in entries:
            try:
                mt_lang, _ = lang_model.predict(entry.get("mt_adapter", ""), k=1)
                ref_lang, _ = lang_model.predict(entry.get("ref", ""), k=1)
            except Exception as e:
                print(f"언어 감지 중 오류 발생 (key={key}): {e}")
                invalid_combinations.add((
                    entry.get("file_index"),
                    entry.get("translation_type"),
                    entry.get("dataset_size")
                ))
                continue

            if mt_lang[0] != ref_lang[0]:
                print(f"Mismatch Detected: File_Index={entry.get('file_index')}, Translation_Type={entry.get('translation_type')}, Dataset_Size={entry.get('dataset_size')}")
                invalid_combinations.add((
                    entry.get("file_index"),
                    entry.get("translation_type"),
                    entry.get("dataset_size")
                ))
    
    return invalid_combinations

def main():
    # 파일 경로 설정
    parser = argparse.ArgumentParser(description="Visualize and compare Comet scores from Adapter and Gemma results.")
    parser.add_argument('--json_file', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_csv_file', type=str, required=True, help="Path to the input JSON file.")
    args = parser.parse_args()
    lang_model_path = "/disk/NIA_POC2/sjkim_folder/lid.176.ftz"
    try:
        # JSON 데이터 로드
        data = load_json(args.json_file)
        
        # FastText 모델 로드
        lang_model = load_fasttext_model(lang_model_path)
        
        # 유효하지 않은 조합 탐지
        invalid_combinations = detect_invalid_combinations(data, lang_model)
        
        # 새로운 CSV 파일 생성
        create_updated_csv(args.csv_file, invalid_combinations, args.output_csv_file)
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
