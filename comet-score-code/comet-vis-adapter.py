import pandas as pd
import matplotlib.pyplot as plt
import argparse

# CSV 파일을 읽고 DataFrame으로 변환하는 함수
def load_csv(file_path):
    return pd.read_csv(file_path)

# 두 CSV 파일을 비교하여 Comet Score를 시각화하고 값을 CSV로 저장하는 함수 (Translation_Type별로 별도 그래프)
def visualize_comet_scores_comparison(csv_file_adapter, csv_file_gemma, output_dir):
    # Step 1: CSV 파일을 각각 읽어 DataFrame으로 변환
    df_adapter = load_csv(csv_file_adapter)
    df_gemma = load_csv(csv_file_gemma)

    # Step 2: Dataset_Size 200, 400, 800, 1600만 필터링
    valid_dataset_sizes = [200, 400, 800, 1600]
    df_adapter = df_adapter[df_adapter['Dataset_Size'].isin(valid_dataset_sizes)]
    df_gemma = df_gemma[df_gemma['Dataset_Size'].isin(valid_dataset_sizes)]

    # Step 3: Translation_Type과 Dataset_Size별로 그룹화하고 각각 Comet_Score_mt_adapter, Comet_Score_mt_gemma 평균 계산
    grouped_adapter = df_adapter.groupby(['Translation_Type', 'Dataset_Size'])['Comet_Score_mt_adapter'].mean().reset_index()
    grouped_gemma = df_gemma.groupby(['Translation_Type', 'Dataset_Size'])['Comet_Score_mt_gemma'].mean().reset_index()

    # 두 데이터프레임을 합쳐서 하나의 데이터프레임으로 만들기
    merged_df = pd.merge(grouped_adapter, grouped_gemma, on=['Translation_Type', 'Dataset_Size'], how='inner')

    # 합친 데이터프레임을 CSV 파일로 저장
    merged_csv_path = f'{output_dir}/comet_scores_adapter_20241015.csv'
    merged_df.to_csv(merged_csv_path, index=False)

    print(f"Comet scores comparison saved to {merged_csv_path}")

    # Step 4: Translation_Type별로 각각의 그래프 생성
    for translation_type in grouped_adapter['Translation_Type'].unique():
        plt.figure(figsize=(10, 6))

        # Adapter 데이터 필터링
        subset_adapter = grouped_adapter[grouped_adapter['Translation_Type'] == translation_type]
        # Gemma 데이터 필터링
        subset_gemma = grouped_gemma[grouped_gemma['Translation_Type'] == translation_type]

        # Adapter 결과 실선으로 그리기
        plt.plot(subset_adapter['Dataset_Size'], subset_adapter['Comet_Score_mt_adapter'], 
                 marker='o', linestyle='-', label='Adapter', color='blue')

        # Adapter 점에 수치 추가
        for i, value in enumerate(subset_adapter['Comet_Score_mt_adapter']):
            plt.text(subset_adapter['Dataset_Size'].iloc[i], value, f'{value:.10f}', ha='center', va='bottom', fontsize=10)

        # Gemma 결과 점선으로 그리기
        plt.plot(subset_gemma['Dataset_Size'], subset_gemma['Comet_Score_mt_gemma'], 
                 marker='o', linestyle='--', label='Gemma', color='red')

        # Gemma 점에 수치 추가
        for i, value in enumerate(subset_gemma['Comet_Score_mt_gemma']):
            plt.text(subset_gemma['Dataset_Size'].iloc[i], value, f'{value:.10f}', ha='center', va='bottom', fontsize=10)

        # 그래프 설정
        plt.title(f'Comparison of Comet Scores for {translation_type}')
        plt.xlabel('Dataset Size')
        plt.ylabel('Average Comet Score')
        plt.legend(title='Model Type')
        plt.grid(True)
        plt.xticks(valid_dataset_sizes)  # X축을 dataset size로 고정

        # 그래프 저장 및 보여주기
        plt.tight_layout()
        output_graph_path = f'{output_dir}/comet_score_comparison_{translation_type}_adapter.png'
        plt.savefig(output_graph_path)  # PNG 파일로 저장
        print(f"Graph for {translation_type} saved to {output_graph_path}")

# argparse를 사용하여 입력 파일과 출력 디렉토리 경로를 받음
def main():
    parser = argparse.ArgumentParser(description="Visualize and compare Comet scores from Adapter and Gemma results.")
    parser.add_argument('--csv_file_adapter', type=str, required=True, help="Path to the CSV file for Adapter results.")
    parser.add_argument('--csv_file_gemma', type=str, required=True, help="Path to the CSV file for Gemma results.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output CSV and graphs.")

    args = parser.parse_args()

    # 결과 시각화 및 저장
    visualize_comet_scores_comparison(args.csv_file_adapter, args.csv_file_gemma, args.output_dir)

if __name__ == "__main__":
    main()
