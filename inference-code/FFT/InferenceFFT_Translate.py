import torch
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import sys
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import json
import time
import datetime
import pandas as pd
# from comet import download_model, load_from_checkpoint

torch.manual_seed(0)

# comet_sum = 0
# G_comet_sum = 0


FT_time = float(0)
G_time = float(0)

# def CalcStatF(comet):
#     global comet_sum
#     comet_sum += comet

# def CalcStatG(comet):
#     global G_comet_sum
#     G_comet_sum += comet
    
# comet_model_path = download_model("Unbabel/XCOMET-XL")
# comet_model = load_from_checkpoint(comet_model_path)

dataset_size = sys.argv[1]
if (int(dataset_size) == 0):
    dataset_size = "ALL"
    
test_num = int(sys.argv[2])
if (test_num == 0):
    test_num = 1350
    

dataset_type = sys.argv[3]

tans_type = sys.argv[4]

is_gemma_train = sys.argv[5]

BASE_MODEL = "google/gemma-2b-it"
# FINETUNE_MODEL = "./exper/MergeModel/Translate-gemma-2b-it-sum-ko.Rank" + str(rank) + "." + dataset_size
FINETUNE_MODEL = "./exper/FullFTModel/FFTModel-Translate." + tans_type + "." + dataset_size


model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"":0})
finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)

pipe_finetuned = pipeline("text-generation", model=finetune_model, tokenizer=tokenizer, max_new_tokens=512)

# test_dataset = "dataset-500"

test_dataset2 = 'datasetTranslate' + dataset_type + '-' + tans_type + '-' + str(int(int(dataset_size)/2))
test_dataset3 = 'datasetTranslate' + str(int(dataset_type)+1) + '-' + tans_type + '-' + str(int(int(dataset_size)/2))

dataset2 = load_from_disk(test_dataset2)
dataset3 = load_from_disk(test_dataset3)

dataset = DatasetDict({
    'train': Dataset.from_pandas(pd.concat([dataset2['train'].to_pandas(), dataset3['train'].to_pandas()], ignore_index=True)),
    'validation': Dataset.from_pandas(pd.concat([dataset2['validation'].to_pandas(), dataset3['validation'].to_pandas()], ignore_index=True))
})


doc = dataset['validation']

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

rouge = Rouge()

dest_lan = ''
if tans_type[2:] == 'ko':
    dest_lan = '한국어'
elif tans_type[2:] == 'en':
    dest_lan = '영어'
elif tans_type[2:] == 'cn':
    dest_lan = '중국어'
elif tans_type[2:] == 'jp':
    dest_lan = '일본어'

for i in range(test_num):
    if tans_type == 'kocn':
        messages = [
            {
                "role": "user",
                "content": "请将以下文章翻译成中文:\n\n{}".format(doc['src'][i])
            }
        ]
    else:
        messages = [
        {
            "role": "user",
            "content": "다음 글을 {}로 번역해주세요:\n\n{}".format(dest_lan, doc['src'][i])
        }
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # print(prompt)
    FT_start = time.time()
    outputs = pipe_finetuned(
        prompt,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        add_special_tokens=True,
        repetition_penalty=1.3,
        # no_repeat_ngram_size=5
    )
    FT_end = time.time()
    
    FT_sec = (FT_end - FT_start)
    FT_time += FT_sec
    
    Fine_tuned_out = outputs[0]["generated_text"][len(prompt):]

    if int(is_gemma_train) == 1:
        G_start = time.time()
        outputs_gemma = pipe(
            prompt,
            do_sample=True,
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            add_special_tokens=True
        )
        G_end = time.time()
        
        G_sec = (G_end - G_start)
        G_time += G_sec
        
        Gemma_out = outputs_gemma[0]["generated_text"][len(prompt):]
        Gemma_file_name = "./exper/gemma_out/Gemma_output_" + tans_type + "_" + str(i) + ".txt"
        Gemma_file = open(Gemma_file_name, "w")
        Gemma_file.write(Gemma_out)
        Gemma_file.close()
        Gemma_time_name = "./exper/gemma_out/Gemma_time_" + tans_type + "_" + str(i) + ".txt"
        Gemma_time = open(Gemma_time_name, "w")
        Gemma_time.write(str(G_sec))
        Gemma_time.close()
    
    else:
        Gemma_file_name = "./exper/gemma_out/Gemma_output_" + tans_type + "_" + str(i) + ".txt"
        Gemma_file = open(Gemma_file_name, "r")
        Gemma_out = Gemma_file.read()
        Gemma_file.close()
        
        Gemma_time_name = "./exper/gemma_out/Gemma_time_" + tans_type + "_" + str(i) + ".txt"
        Gemma_time = open(Gemma_time_name, "r")
        G_sec = float(Gemma_time.read())
        Gemma_time.close()
        G_time += G_sec
    
    reference = doc['dest'][i]
    
    # FT_comet_data = []
    # FT_comet_data.append({
    #     "src": doc['src'][i],
    #     "mt": Fine_tuned_out,
    #     "ref": reference
    # })
    
    # Gemma_comet_data = []
    # Gemma_comet_data.append({
    #     "src": doc['src'][i],
    #     "mt": Gemma_out,
    #     "ref": reference
    # })
    
    # FT_model_output = comet_model.predict(FT_comet_data, batch_size=8, gpus=1)
    # Gemma_model_output = comet_model.predict(Gemma_comet_data, batch_size=8, gpus=1)
    
    output_file = "./exper/InferenceResultFFT/Translate_" + tans_type + "_" + dataset_size + "_" + str(i) + ".out"
    f = open(output_file, "w")
    f.write("Input_file :\n")
    f.write(prompt)
    f.write("\n\nAdapter Fine-tuned Output :\n")
    f.write(Fine_tuned_out)
    # f.write("\nAdapter Fine-tuning COMET Score -- ")
    # FT_comet = FT_model_output.scores
    # f.write(str(FT_comet[0]))
    # CalcStatF(FT_comet[0])
    f.write("\n\nGemma Output :\n")
    f.write(Gemma_out)
    # f.write("\nGemma Model COMET Score -- ")
    # G_comet = Gemma_model_output.scores
    # f.write(str(G_comet[0]))
    # CalcStatG(G_comet[0])
    f.write("\n\nLabel trans result :\n")
    f.write(reference)
    f.close()
    
    # print("OUTPUT ", i)
    # print(outputs[0]["generated_text"][len(prompt):])
    # print()

avg_FT_time = FT_time/test_num
avg_G_time = G_time/test_num
# avg_FT_comet = comet_sum/test_num
# avg_G_comet = G_comet_sum/test_num

output_report = "./exper/InferenceReportFFT/Translate_Inference_" + tans_type + "_" + dataset_size + ".out"
f2 = open(output_report, "w")
f2.write("Dataset: Translate\n")
f2.write("Test Dataset Size: " + str(test_num) + "\n")
f2.write("Fine-tuning Inference Time: " + str(avg_FT_time) + "\n")
f2.write("Gemma Model Inference Time: " + str(avg_G_time) + "\n")
# f2.write("\nFine-tuning Score:\n")
# f2.write("COMET: " + str(avg_FT_comet) + "\n")
# f2.write("\nGemma Model Score\n")
# f2.write("COMET: " + str(avg_G_comet) + "\n")
f2.close()