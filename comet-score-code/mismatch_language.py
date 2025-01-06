import json
import csv
import argparse
import os
import random
import fasttext

# Set the seed for reproducibility
seed = 4
random.seed(seed)

def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print("JSON 데이터 로드 성공, 데이터 타입:", type(data))

        # 최상위 키 값이 딕셔너리인 경우 처리
        if isinstance(data, dict):
            combined_data = []
            for key, value in data.items():
                if isinstance(value, list):  # 각 값이 리스트일 경우 추가
                    combined_data.extend(value)
            print(f"총 {len(combined_data)}개의 항목을 로드했습니다.")
            return combined_data
        
        # JSON 파일이 이미 리스트 형태일 경우
        elif isinstance(data, list):
            print("JSON 데이터가 리스트 형태입니다.")
            return data
        else:
            raise ValueError("JSON 데이터가 리스트나 딕셔너리 형태가 아닙니다.")
# Dataset별 언어 불일치 비율 계산 함수
def calculate_miss_rate_by_dataset_and_type(lang_model, data, key):
    miss_rates_by_type = {}
    
    for entry in data:
        translation_type = entry.get("translation_type", "Unknown")
        dataset_size = entry.get('dataset_size', 'Unknown')
        
        if translation_type not in miss_rates_by_type:
            miss_rates_by_type[translation_type] = {}
        
        if dataset_size not in miss_rates_by_type[translation_type]:
            miss_rates_by_type[translation_type][dataset_size] = {"total": 0, "mismatch": 0}
        
        # 언어 감지
        try:
            mt_lang, mt_conf = lang_model.predict(entry[key], k=1)
            ref_lang, ref_conf = lang_model.predict(entry["ref"], k=1)
        except Exception as e:
            print(f"Error detecting language for entry {entry}: {str(e)}")
            continue
        
        miss_rates_by_type[translation_type][dataset_size]["total"] += 1
        if mt_lang[0] != ref_lang[0]:
            miss_rates_by_type[translation_type][dataset_size]["mismatch"] += 1

    # Miss율 계산
    for translation_type, datasets in miss_rates_by_type.items():
        for size, counts in datasets.items():
            total = counts["total"]
            mismatch = counts["mismatch"]
            miss_rates_by_type[translation_type][size]["miss_rate"] = (mismatch / total) * 100 if total > 0 else 0.0

    return miss_rates_by_type

# CSV로 저장하는 함수
def save_miss_rates_by_type_to_csv(miss_rates_by_type, output_file):
    fieldnames = ['Translation_Type', 'Dataset_Size', 'Total_Entries', 'Mismatch_Count', 'Miss_Rate']
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for translation_type, datasets in miss_rates_by_type.items():
            for size, stats in datasets.items():
                writer.writerow({
                    "Translation_Type": translation_type,
                    "Dataset_Size": size,
                    "Total_Entries": stats["total"],
                    "Mismatch_Count": stats["mismatch"],
                    "Miss_Rate": stats["miss_rate"]
                })

def main():
    # argparse를 사용하여 입력 파일과 출력 파일 경로 등을 받음
    parser = argparse.ArgumentParser(description="Calculate language mismatch rates by dataset size and translation type.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument('--is_adpater', type=bool, default=False)
    args = parser.parse_args()

    # 언어 모델 로드
    lang_model = fasttext.load_model('/disk/NIA_POC2/sjkim_folder/lid.176.ftz')

    # JSON 데이터를 불러오기
    data = load_json(args.input_file)

    # Translation Type별 Miss율 계산
    if args.is_adpater == True : 
        miss_rates_by_type = calculate_miss_rate_by_dataset_and_type(lang_model, data, 'mt_adapter')
    else : 
        miss_rates_by_type = calculate_miss_rate_by_dataset_and_type(lang_model, data, 'mt_gemma')

    # CSV로 저장
    save_miss_rates_by_type_to_csv(miss_rates_by_type, args.output_file)

if __name__ == "__main__":
    main()
