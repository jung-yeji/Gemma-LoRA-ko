import json
import csv
import argparse
from comet import download_model, load_from_checkpoint
import os
import random
from langdetect import detect  # 언어 식별 라이브러리
import fasttext
# Set the seed for reproducibility
seed = 4
random.seed(seed)



# Step 1: 모델을 다운로드하고 체크포인트를 로드합니다.

#def load_comet_model(model_name="Unbabel/XCOMET-XL"):
def load_comet_model(model_name="Unbabel/wmt22-comet-da"):
    model_path = download_model(model_name)
    return load_from_checkpoint(model_path)

# JSON 데이터를 불러오는 함수
def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_comet_scores(lang_model, comet_model, data, key, batch_size=8, use_gpu=True):
    
    comet_data = []

    # COMET 모델 입력 데이터 구성
    for entry in data:
        comet_data.append({"src": entry["src"], "mt": entry[key], "ref": entry["ref"]})

    # 모델을 사용하여 점수 예측
    gpus = 1 if use_gpu else 0
    # COMET 모델에서 점수 예측
    model_output = comet_model.predict(comet_data, batch_size=batch_size, gpus=gpus)
    return model_output["scores"]  # 언어 감지 생략, 모든 점수 반환


# CSV로 저장하는 함수
def save_scores_to_csv(lang_model, data, output_file, key, model, batch_size=8, use_gpu=True):
    # CSV 필드명 설정
    fieldnames = ['Rank', 'File_Index', 'Translation_Type', 'Dataset_Size', f'Comet_Score_{key}']

    with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for rank, entries in data.items():

            if not entries:
                continue  # Dataset_Size가 400인 항목이 없으면 건너뜀

            # Comet 스코어 계산
            try:
                rank_scores = calculate_comet_scores(lang_model, model, entries, key, batch_size=batch_size, use_gpu=use_gpu)
            except Exception as e:
                print(f"Error calculating scores for rank {rank}: {str(e)}")
                continue

            # 데이터를 CSV에 저장
            for entry, score in zip(entries, rank_scores):
                writer.writerow({
                    'Rank': rank,
                    'File_Index': entry['file_index'],
                    'Translation_Type': entry['translation_type'],  # 번역 유형 추가
                    'Dataset_Size': entry['dataset_size'],  # 데이터셋 크기 추가
                    f'Comet_Score_{key}': score
                })

def main():
    # argparse를 사용하여 입력 파일과 출력 파일 경로 등을 받음
    parser = argparse.ArgumentParser(description="Comet score calculation for translation results.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_csv_adapter', type=str, required=True, help="Path to the output CSV file for Adapter results.")
    parser.add_argument('--output_csv_gemma', type=str, required=True, help="Path to the output CSV file for Gemma results.")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for Comet model predictions on Adapter results.")
    parser.add_argument('--use_gpu', type=bool, default=True, help="Whether to use GPU for Comet model predictions.")

    args = parser.parse_args()

    # Step 2: 모델을 불러옵니다.
    model = load_comet_model()
    lang_model = fasttext.load_model('/disk/NIA_POC2/sjkim_folder/lid.176.ftz')

    # Step 3: JSON 데이터를 불러오고 Adapter와 Gemma 결과에 대한 Comet 스코어 계산 후 CSV 저장
    data = load_json(args.input_file)

    # Adapter 결과에 대한 Comet 스코어 계산 후 CSV 저장
    save_scores_to_csv(lang_model, data, args.output_csv_adapter, 'mt_adapter', model, batch_size=args.batch_size, use_gpu=args.use_gpu)
    # Gemma 결과에 대한 Comet 스코어 계산 후 CSV 저장
    save_scores_to_csv(lang_model, data, args.output_csv_gemma, 'mt_gemma', model, batch_size=args.batch_size, use_gpu=args.use_gpu)

if __name__ == "__main__":
    main()
