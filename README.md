# Gemma-LoRA-ko

## 1. 개요
- 온디바이스용 소형 언어모델인 [Gemma 2B](https://huggingface.co/google/gemma-2b) 사전학습 모델을 한국어 데이터셋으로 파인튜닝하는 코드를 제공합니다.
- **FFT (Full Parameter Fine-tuning)**, **Adapter ([LoRA](https://arxiv.org/abs/2106.09685))** 두 가지 버전을 지원합니다.
- 파인튜닝으로 개선된 정확도를 평가하기 위한 매트릭인 [ROUGE](https://www.microsoft.com/en-us/research/publication/rouge-a-package-for-automatic-evaluation-of-summaries/), [BLEU](https://dl.acm.org/doi/10.3115/1073083.1073135), [COMET](https://github.com/Unbabel/COMET) 점수 계산이 포함되어 있습니다.
- 이 연구에서 사용된 파인튜닝 파라미터와 한국어 데이터셋 정보는 아래 표와 같습니다.
  - 데이터셋은 코드에 포함되어 있지 않으며, [AI 허브](https://www.aihub.or.kr/)에서 무료로 다운로드 받을 수 있습니다.

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>분야</th>
      <th>Type</th>
      <th>Dataset size</th>
      <th>Learning rate</th>
      <th>Epoch</th>
      <th>Batch</th>
      <th>AI 허브 데이터셋</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="12">번역</td>
      <td rowspan="2">한영</td>
      <td>FFT</td>
      <td>400</td>
      <td>5.00E-06</td>
      <td>20</td>
      <td>20</td>
      <td rowspan="4">일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터 / 방송콘텐츠 한국어-영어 번역 말뭉치</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>400</td>
      <td>2.00E-04</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td rowspan="2">영한</td>
      <td>FFT</td>
      <td>1600</td>
      <td>5.00E-06</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>1600</td>
      <td>2.00E-04</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td rowspan="2">한중</td>
      <td>FFT</td>
      <td>400</td>
      <td>5.00E-06</td>
      <td>20</td>
      <td>20</td>
      <td rowspan="8">방송콘텐츠 한국어-아시아어 번역 말뭉치 / 방송 콘텐츠 한-중, 한-일 번역 병렬 말뭉치</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>400</td>
      <td>5.00E-04</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td rowspan="2">중한</td>
      <td>FFT</td>
      <td>400</td>
      <td>5.00E-06</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>400</td>
      <td>2.00E-04</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td rowspan="2">한일</td>
      <td>FFT</td>
      <td>400</td>
      <td>5.00E-06</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>400</td>
      <td>2.00E-04</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td rowspan="2">일한</td>
      <td>FFT</td>
      <td>400</td>
      <td>5.00E-06</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>400</td>
      <td>2.00E-04</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td rowspan="8">요약</td>
      <td rowspan="2">신문기사</td>
      <td>FFT</td>
      <td>1600</td>
      <td>1.00E-07</td>
      <td>4</td>
      <td>20</td>
      <td rowspan="6">문서요약 텍스트</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>1600</td>
      <td>6.00E-05</td>
      <td>4</td>
      <td>20</td>
    </tr>
    <tr>
      <td rowspan="2">기고문/잡지</td>
      <td>FFT</td>
      <td>400</td>
      <td>6.00E-07</td>
      <td>4</td>
      <td>20</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>400</td>
      <td>3.00E-04</td>
      <td>4</td>
      <td>20</td>
    </tr>
    <tr>
      <td rowspan="2">법률문서</td>
      <td>FFT</td>
      <td>400</td>
      <td>5.00E-06</td>
      <td>4</td>
      <td>20</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>400</td>
      <td>2.00E-04</td>
      <td>4</td>
      <td>20</td>
    </tr>
    <tr>
      <td rowspan="2">발언</td>
      <td>FFT</td>
      <td>1600</td>
      <td>5.00E-07</td>
      <td>4</td>
      <td>20</td>
      <td rowspan="2">채용면접 인터뷰 데이터</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>1600</td>
      <td>6.00E-05</td>
      <td>4</td>
      <td>20</td>
    </tr>
    <tr>
      <td rowspan="4">질의응답</td>
      <td rowspan="2">대화</td>
      <td>FFT</td>
      <td>1600</td>
      <td>5.00E-06</td>
      <td>20</td>
      <td>20</td>
      <td rowspan="2">한국어 대화 요약</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>1600</td>
      <td>5.00E-04</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td rowspan="2">자연어</td>
      <td>FFT</td>
      <td>400</td>
      <td>5.00E-06</td>
      <td>4</td>
      <td>20</td>
      <td rowspan="2">한국어 성능이 개선된 초거대AI 언어모델 개발 데이터</td>
    </tr>
    <tr>
      <td>Adapter</td>
      <td>400</td>
      <td>2.00E-04</td>
      <td>4</td>
      <td>20</td>
    </tr>
  </tbody>
</table>



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
- AI 허브에 공개된 한국어 데이터셋을 파인튜닝 학습용으로 가공
- 데이터셋 종류는 **QA**, **Summary**, **Translate**로 나뉩니다.

### **MakeModel-code**
- 파인튜닝 모델을 만드는 코드입니다.
- **FFT(Full Parameter Fine-Tuning)** 은 파인튜닝된 모델이 저장
- **Adapter(LoRA)** 는 모델과 별도로 어댑터 파일이 저장됩니다. (사전학습 모델과 결합 필요)
- 데이터셋 종류에 따라 **QA**, **Summary**, **Translate**로 코드가 구분됩니다.

### **merge-code**
- 사전학습 모델에 **Adapter**를 결합
- 데이터셋 종류에 따라 **QA**, **Summary**, **Translate**로 코드가 구분됩니다.

### **inference-code**
- 추론 수행
- **FFT**와 **Adapter** 두 가지 버전을 지원합니다.
- 데이터셋 종류에 따라 **QA**, **Summary**, **Translate**로 코드가 구분됩니다.

### **comet-score-code**
- 추론 결과를 분석하여 **Comet score**를 계산
- **Translate** 결과에만 사용됩니다.
- 번역이 실패 경우(원하는 언어로 답변이 나오지 않았을 때), 언어 불일치를 인식하고 0점 부여



## 3. 실행 방법

### **실험 환경**
**Python**: 3.9 버전

### **초기 설정**
필요한 Python 라이브러리를 설치하려면 다음 명령어를 실행하세요:
```bash
pip install -r Gemma-LoRA-ko/requirements.txt
```
참고: PyTorch와 같은 라이브러리는 GPU 환경에 따라 버전을 맞춰야 합니다.



### **Dataset-gen-code**
- 학습용 데이터를 생성하는 코드입니다.
- **QA** 데이터셋 생성 예시:
  ```bash
  python ./gen_dataset_QA.py <max_files> <max_files_val>
  ```
  - `max_files`: 학습 데이터 파일 수 (int).  
  - `max_files_val`: 검증 데이터 파일 수 (int). (본 과제에서는 100으로 고정)



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
   - 번역 요청 언어와 출력 언어의 불일치를 감지하고 결과를 업데이트:
     ```bash
     python mismatch_language.py --input_file <output>.json \
     --output_file mismatching/test-fft.csv \
     --is_adapter True
     ```
   - 언어 불일치 시 0점 부여


## 4. 예제
각 폴더 내 `example.sh` 파일에서 실행 예제와 흐름을 확인할 수 있습니다.  
자세한 사용 방법은 코드 주석을 참고하세요.


## 5. 정보
- 이 코드는 2024년도 정부(과학기술정보통신부)의 재원으로 **한국지능정보사회진흥원**의 지원을 받아 **대구경북과학기술원**이 수행한 연구의 소스 코드입니다(2024년 데이터 활용 기획검증 사업 - 온디바이스 생성모델 파인튜닝 및 추론)
- 2024 한국소프트웨어종합학술대회(KSC2024) 논문 게재 (제목: 온디바이스 생성모델 파인튜닝 및 추론 성능 분석)
- 작성자 연락처: jung.yeji@dgist.ac.kr

