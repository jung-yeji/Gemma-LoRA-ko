# Gemma-LoRA-ko

**Updated at: 2024.12.04**

## 1. 개요

이 코드는 대구경북과학기술원(DGIST)에서 수행한 **"2024년 데이터 활용 기획·검증 사업 - 온디바이스 생성모델 파인튜닝 및 추론"** 과제의 원본 소스코드 산출물입니다.  
저작권 등의 문제로 인해 데이터셋 원본 파일들은 포함되어 있지 않습니다.

---

## 2. 코드 구조

코드는 다음과 같은 구조로 구성되어 있습니다:

```
Gemma-LoRA-ko
├─ comet-score-code
├─ Dataset-gen-code
├─ inference-code
│   ├─ Adapter
│   └─ FFT
├─ MakeModel-code
│   ├─ Adapter
│   └─ FFT
└─ merge-code
```

각 폴더의 목적은 다음과 같습니다:

### **Dataset-gen-code**
- 학습용 데이터셋을 생성하는 코드입니다.
- 데이터셋 종류는 **QA**, **Summary**, **Translate**로 나뉩니다.

### **MakeModel-code**
- 최초 모델을 생성하는 코드입니다.
- **FFT**와 **Adapter (LoRA)** 두 가지 버전을 지원합니다.
- 데이터셋 종류에 따라 **QA**, **Summary**, **Translate**로 구분됩니다.

### **merge-code**
- 생성된 모델에 Adapter를 결합하는 코드입니다.
- **Adapter** 버전만 지원됩니다.
- 데이터셋 종류에 따라 **QA**, **Summary**, **Translate**로 구분됩니다.

### **inference-code**
- 추론을 수행하는 코드입니다.
- **FFT**와 **Adapter (LoRA)** 두 가지 버전을 지원합니다.
- 데이터셋 종류에 따라 **QA**, **Summary**, **Translate**로 구분됩니다.

### **comet-score-code**
- 추론 결과를 분석하여 **Comet score**를 계산하는 코드입니다.
- **Translate** 결과에만 사용됩니다.

---

## 3. 실행 방법

### **실험 환경**
본 과제의 모든 실험은 아래와 같은 환경에서 진행되었습니다:
- **OS**: Ubuntu 22.04 (Linux Kernel 6.8.0-40-generic)
- **CPU**: AMD EPYC 9354 32-Core Processor × 2
- **GPU**: NVIDIA A100 × 4
- **메모리**: 512GB DRAM
- **Python**: 3.9 버전

### **초기 설정**
필요한 Python 라이브러리를 설치하려면 다음 명령어를 실행하세요:
```bash
pip install -r Gemma-LoRA-ko/requirements.txt
```
참고: PyTorch와 같은 라이브러리는 GPU 환경에 따라 버전을 맞춰야 합니다.

---

### **Dataset-gen-code**
- 학습용 데이터를 생성하는 코드입니다.
- **QA** 데이터셋 생성 예시:
  ```bash
  python ./gen_dataset_QA.py <max_files> <max_files_val>
  ```
  - `max_files`: 학습 데이터 파일 수 (int).  
  - `max_files_val`: 검증 데이터 파일 수 (int). (본 과제에서는 100으로 고정)

---

### **MakeModel-code**
- FFT 또는 Adapter(LoRA)를 사용하여 모델을 생성합니다.
- **QA**에서 Adapter 사용 예시:
  ```bash
  python3 makeLoRA_QA.py <dataset_size> <rank>
  ```
  - `dataset_size`: `gen_dataset_<dataset_type>.py`로 생성된 데이터셋의 `max_files` 값과 일치해야 합니다 (int).  
  - `rank`: Adapter(LoRA)의 Rank 크기 (int).  

**Translate**의 경우 추가 인자가 필요합니다:
```bash
python3 makeLoRA_Translate.py <dataset_size> <rank> <dataset_type> <trans_type>
```
- `dataset_type`: 데이터셋의 타입(폴더명과 일치).  
- `trans_type`: 번역 방향 (e.g., `enko`, `cnko`, `jpko`, 등).

---

### **merge-code**
- 생성된 모델과 Adapter를 결합하는 코드입니다.
- **QA**에서 사용 예시:
  ```bash
  python3 mergeModel_QA.py <dataset_size> <rank>
  ```
**Translate**에서 사용 예시:
```bash
python3 mergeModel_Translate.py <dataset_size> <rank> <dataset_type> <trans_type>
```

---

### **comet-score-code**
Translate 결과를 분석하여 **Comet score**를 계산합니다. 실행 순서는 다음과 같습니다:

1. **번역 결과 변환**:
   ```bash
   python convert-fft.py --input_folder <input_folder> --output_file <output>.json
   ```
   또는
   ```bash
   python convert-adapter.py --input_folder <input_folder> --output_file <output>.json
   ```
   - 추론 결과 폴더를 입력으로 사용하여 결과를 JSON 파일로 정리합니다.

2. **Comet 점수 계산**:
   ```bash
   python comet-score.py --input_file <output>.json \
   --output_csv_adapter <output_results_adapter.csv> \
   --output_csv_gemma <output_results_gemma.csv> \
   --batch_size 10 \
   --use_gpu True
   ```

3. **결과 시각화**:
   ```bash
   python comet-vis-fft.py --csv_file_adapter <output_results_adapter.csv> \
   --csv_file_gemma <output_results_gemma.csv> \
   --output_dir "./"
   ```

4. **언어 매칭 확인**:
   - 언어 불일치를 감지하고 결과를 업데이트:
     ```bash
     python mismatch_language.py --input_file <output>.json \
     --output_file mismatching/test-fft.csv \
     --is_adapter True
     ```

---

### **예제**
각 폴더 내 `example.sh` 파일에서 실행 예제와 흐름을 확인할 수 있습니다.  
자세한 사용 방법은 코드 주석을 참고하세요.
