updated at 2024.12.04
1. Overview
이 코드는 대구경북과학기술원(DGIST)에서 수행한 ‘2024년 데이터 활용 기획·검증 사업 - 온디바이스 생성모델 파인튜닝 및 추론’ 과제의 원
본 소스코드 산출물입니다. 저작권 등의 문제로 인해 이 코드에는 데이터셋 원본 파일들은 포함되어 있지 않습니다.
2.
코드 전체 구조
전체 구조는 다음과 같습니다.
nia-poc-codes
├─comet-score-code
├─Dataset-gen-code
├─inference-code
│ ├─Adapter
│ └─FFT
├─MakeModel-code
│ ├─Adapter
│ └─FFT
└─merge-code
라는 최상위 폴더를 기준으로 5개 종류의 코드 구성으로 되어 있습니다.
실행하는 순서를 고려해 정렬한 뒤, 각 폴더들 내 코드들의 목적을 설명하면 다음과 같습니다.
nia-poc-codes
Dataset-gen-code
의 json 파일들을 가지고 와 학습용 Dataset을 생성하는 코드들입니다.
데이터셋에 따라 QA / Summary / Translate 로 분류됩니다.
Dataset
MakeModel-code
최초 모델을 생성하는 코드입니다.
FFT / Adapter(LoRA)용, 2가지 버전이 있습니다.
데이터셋에 따라 QA / Summary / Translate 로 분류됩니다.
merge-code
모델 생성 이후, Adapter와 결합하는 코드입니다.
즉 Adapter를 위한 버전만 존재합니다.
데이터셋에 따라 QA / Summary / Translate 로 분류됩니다.
inference-code
추론을 위한 코드들입니다.
FFT / Adapter(LoRA)용, 2가지 버전이 있습니다.
데이터셋에 따라 QA / Summary / Translate 로 분류됩니다.
comet-score-code
추론의 결과물을 분석하는 과정에서 Comet score 계산을 위한 코드들입니다.
오직 Translate 결과들에만 사용되어야 합니다.
NIA POC
명세서
13.
실행 방법
과제 수행 환경
모든 실험은 다음과 같은 환경에서 진행되었습니다.
- Linux Kernel 6.8.0-40-generic
- Ubuntu 22.04
- AMD EPYC 9354 32-Core Processor * 2
- A100 * 4
- 512GB DRAM
- Python 3.9
초기 설정
대부분의 코드들은 python library에 의한 의존성이 있습니다.
이를 해결하기 위해 다음과 같은 명령어를 입력해야 합니다.
pip install -r nia-poc-codes\requirements.txtrequirements.txt
대부분의 의존성을 해결해줄 수 있지만, Pytorch와 같은 GPU 의존성이 존재하는 경우, 이를 고려해 Pytorch버전을 맞춰야만 합니다.
Dataset-gen-code
gen_dataset_<dataset_type>.py
내의 코드들은 모두 같은 실행법을 가집니다.
gen_dataset_QA.py 를 예시로 들어보면 다음과 같습니다.
Dataset-gen-code
python ./gen_dataset_QA.py <max_files> <max_files_val>
의 수 ( int )
max_files_val : Validation data의 수 ( int )
본 과제에선 max_files_val을 100으로 고정했습니다.
max_files : Traning data
MakeModel-code
코드들은 FFT, Adapter 폴더로 나누어져 있으나 기본적인 사용방법은 똑같습니다.
다만 번역의 경우 한개의 인자가 추가됩니다.
makeLoRA_QA.py를 예시로 설명하겠습니다.
python3 makeLoRA_QA.py <dataset_size> <rank>
를 통해 만든 dataset의 max_files와 일치시켜주셔야 합니다. ( int )
rank : 적용할 Adapter(LoRA)의 Rank 수를 의미합니다. ( int )
makeLoRA_Translate의 경우 다른 파일들과 다르게 몇가지 인자가 추가됩니다.
dataset_size : gen_dataset_<dataset_type>.py
makeLoRA_Translate.py <dataset_size> <rank> <dataset_type> <tans_type>
를 통해 만든 dataset의 max_files와 일치시켜주셔야 합니다. ( int )
rank : 적용할 Adapter(LoRA)의 Rank 수를 의미합니다. ( int )
dataset_type : Dataset의 Type 구분을 의미합니다. 상세 내용은 코드 보시고 dataset이 존재하는 폴더명과 일치시키면 됩니다.
tans_type : enko / cnko / jpko / koen / kocn / kojp로 구성됩니다. enko의 경우 영→한을 의미합니다. 다른 표기도 같은 맥락으
로 이해하시면 됩니다.
dataset_size : gen_dataset_<dataset_type>.py
NIA POC
명세서
2자세한 실행 방법의 경우 Dataset-gen-code내 example.sh를 참고하시면 됩니다.
merge-code
와 사용법 및 인자가 같습니다. 참고를 위해 적으면 다음과 같습니다.
MakeModel-code
mergeModel_Translate.py <dataset_size> <rank> <dataset_type> <tans_type>
python3 mergeModel_QA.py <dataset_size> <rank>
를 통해 만든 dataset의 max_files와 일치시켜주셔야 합니다. ( int )
rank : 적용할 Adapter(LoRA)의 Rank 수를 의미합니다. ( int )
dataset_type : Dataset의 Type 구분을 의미합니다. 상세 내용은 코드 보시고 dataset이 존재하는 폴더명과 일치시키면 됩니다.
tans_type : enko / cnko / jpko / koen / kocn / kojp로 구성됩니다. enko의 경우 영→한을 의미합니다. 다른 표기도 같은 맥락으
로 이해하시면 됩니다.
더불어 merge-code내 example.sh를 작성해놨으니 참고하시면 됩니다.
dataset_size : gen_dataset_<dataset_type>.py
comet-score-code
코드 실행 순서는 다음과 같습니다.
convert-<type>.py →comet-score.py → comet-vis-<type>.py
convert-<type>.py
<type>
은 fft/adapter로 구성됩니다. 예시는 다음과 같습니다.
python convert-fft.py --input_folder <input_folder> --output_file <ouutput>.json
python convert-adapter.py --input_folder <input_folder> --output_file <ouutput>.json
에 추론 결과 폴더를 넣으면 Translation 결과들을 골라낸 다음, 결과들을 --output_file에 지정된 json으로 정리합니
input folder
다.
comet-score.py
python comet-score.py --input_file "output.json" \
--output_csv_adapter "output_results_adapter.csv" \
--output_csv_gemma "output_results_gemma.csv" \
--batch_size 10 \
--use_gpu True
을 통해 정리된 json파일을 입력으로 넣고,
나머지 adapter / gemma output에 commet 점수를 획득 후 적습니다.
환경에 따라 batch size, GPU 사용 여부를 변경해야 할 수 있습니다.
convert-<type>.py
comet-vis-<type>.py
<type>
은 fft/adapter로 구성됩니다. 예시는 다음과 같습니다.
python comet-vis-fft.py --csv_file_adapter "output_results_adapter.csv" \
--csv_file_gemma "output_results_gemma.csv" \
--output_dir "./"
각각의 결과들을 output 폴더에 시각화 파일 및 CSV 파일을 저장하는 코드입니다.
comet-score.py 를 통해 얻어진 output_results_adapter.csv / output_results_gemma.csv 와 같은 파일을 입력으로 넣고, 자
신이 저장하고 싶은 dir를 지정합니다.
matching.py / matching_gemma.py
NIA POC
명세서
3python matching.py --json_file output.json \
--csv_file output_results_adapter.csv \
--output_csv_file updated_output_results_adapter.csv
python matching_gemma.py --json_file output.json \
--csv_file output_results_gemma.csv \
--output_csv_file updated_output_results_gemma.csv
comet-score
.
계산 결과 이후, convert-<type>.py를 통해 얻은 json을 통해 machine translation 추론 결과와 정답(ref)의 값을
추출합니다
추출된 값들을 각각 언어 감지를 하고, 감지된 언어가 다른 경우 comet-score.py를 통해 얻은 csv파일에서 해당 점수를 0으로 초기
화 합니다.
mismatch_language.py
python mismatch_language.py --input_file output.json \
--output_file mismaching/test-fft.csv
--is_adpater True
matching.py
.
와 비슷하나, mismatch_language.py는 별도로 각 Tranlation Type의 language miss match율을 구하는 코드입
니다
자세한 코드 사용 예시 및 코드 사용 흐름은 comet-score-code내 example.sh에 존재합니다.
