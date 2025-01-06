```markdown
# nia-poc-codes

**Updated at: 2024.12.04**

## 1. Overview

This repository contains the original source code developed for the project **"2024 Data Utilization Planning and Verification Project - On-Device Generative Model Fine-Tuning and Inference"**, conducted at **Daegu Gyeongbuk Institute of Science and Technology (DGIST)**.  
Due to copyright and other restrictions, the dataset's original files are not included.

---
```
## 2. Code Structure

The code is organized into the following structure:
```
nia-poc-codes
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
Each folder serves a specific purpose. Below is a summary of their functions:

### **Dataset-gen-code**
- Scripts for generating training datasets based on input JSON files.
- Dataset types include **QA**, **Summary**, and **Translate**.

### **MakeModel-code**
- Code for creating initial models.
- Supports two versions: **FFT** and **Adapter (LoRA)**.
- Organized into dataset types: **QA**, **Summary**, and **Translate**.

### **merge-code**
- Combines models with adapters after creation.
- Only supports the **Adapter** version.
- Organized into dataset types: **QA**, **Summary**, and **Translate**.

### **inference-code**
- Scripts for running inference using the models.
- Supports two versions: **FFT** and **Adapter (LoRA)**.
- Organized into dataset types: **QA**, **Summary**, and **Translate**.

### **comet-score-code**
- Contains scripts for computing **Comet scores** to analyze inference results.
- Used exclusively for **Translate** results.

---

## 3. Execution Instructions

### **Project Environment**
All experiments were conducted in the following environment:
- **OS**: Ubuntu 22.04 (Linux Kernel 6.8.0-40-generic)
- **CPU**: AMD EPYC 9354 32-Core Processor × 2
- **GPU**: NVIDIA A100 × 4
- **Memory**: 512GB DRAM
- **Python**: Version 3.9

### **Initial Setup**
Install required Python libraries:
```bash
pip install -r nia-poc-codes/requirements.txt
```
Note: Some libraries (e.g., PyTorch) require specific versions based on your GPU environment.

---

### **Dataset-gen-code**
- Generates datasets for training.
- Example command for **QA** dataset:
  ```bash
  python ./gen_dataset_QA.py <max_files> <max_files_val>
  ```
  - `max_files`: Number of training data files (int).  
  - `max_files_val`: Number of validation data files (int). (Set to 100 for this project.)

---

### **MakeModel-code**
- Creates models using either **FFT** or **Adapter (LoRA)**.
- Example command for **QA** using Adapter:
  ```bash
  python3 makeLoRA_QA.py <dataset_size> <rank>
  ```
  - `dataset_size`: Matches the `max_files` value from the dataset generation step (int).  
  - `rank`: Rank of the Adapter (LoRA) applied (int).  

For **Translate**, additional arguments are required:
```bash
python3 makeLoRA_Translate.py <dataset_size> <rank> <dataset_type> <trans_type>
```
- `dataset_type`: Matches the dataset folder name.  
- `trans_type`: Specifies the translation direction (e.g., `enko`, `cnko`, `jpko`, etc.).  

---

### **merge-code**
- Combines models with adapters.
- Example command for **QA**:
  ```bash
  python3 mergeModel_QA.py <dataset_size> <rank>
  ```
For **Translate**, use:
```bash
python3 mergeModel_Translate.py <dataset_size> <rank> <dataset_type> <trans_type>
```

---

### **comet-score-code**
Used to compute **Comet scores** for translation results. Execution flow:
1. **Convert Translation Results**:
   ```bash
   python convert-fft.py --input_folder <input_folder> --output_file <output>.json
   ```
   or
   ```bash
   python convert-adapter.py --input_folder <input_folder> --output_file <output>.json
   ```
   - Extracts translation results and organizes them into JSON format.

2. **Compute Comet Scores**:
   ```bash
   python comet-score.py --input_file <output>.json \
   --output_csv_adapter <output_results_adapter.csv> \
   --output_csv_gemma <output_results_gemma.csv> \
   --batch_size 10 \
   --use_gpu True
   ```

3. **Visualize Results**:
   ```bash
   python comet-vis-fft.py --csv_file_adapter <output_results_adapter.csv> \
   --csv_file_gemma <output_results_gemma.csv> \
   --output_dir "./"
   ```

4. **Language Matching**:
   - Detect language mismatches and update results:
     ```bash
     python mismatch_language.py --input_file <output>.json \
     --output_file mismatching/test-fft.csv \
     --is_adapter True
     ```

---

### **Examples**
- Example scripts and execution flows for each module are provided in `example.sh` files within respective folders.  
- For detailed usage, refer to the comments and documentation within the codebase.

---

If you encounter issues or have questions, feel free to raise them in the Issues tab!
```
