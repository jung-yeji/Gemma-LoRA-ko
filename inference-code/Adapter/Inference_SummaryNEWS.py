import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import sys
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import json
import time
import datetime

torch.manual_seed(0)

rouge_1_r_sum = 0
rouge_1_p_sum = 0
rouge_1_f_sum = 0
rouge_2_r_sum = 0
rouge_2_p_sum = 0
rouge_2_f_sum = 0
rouge_l_r_sum = 0
rouge_l_p_sum = 0
rouge_l_f_sum = 0
bleu_sum = 0

G_rouge_1_r_sum = 0
G_rouge_1_p_sum = 0
G_rouge_1_f_sum = 0
G_rouge_2_r_sum = 0
G_rouge_2_p_sum = 0
G_rouge_2_f_sum = 0
G_rouge_l_r_sum = 0
G_rouge_l_p_sum = 0
G_rouge_l_f_sum = 0
G_bleu_sum = 0

FT_time = float(0)
G_time = float(0)

def CalcStatF(rouge, bleu):
    global rouge_1_r_sum
    global rouge_1_p_sum
    global rouge_1_f_sum
    global rouge_2_r_sum
    global rouge_2_p_sum
    global rouge_2_f_sum
    global rouge_l_r_sum
    global rouge_l_p_sum
    global rouge_l_f_sum
    global bleu_sum
    rouge_1_r_sum += rouge['rouge-1']['r']
    rouge_1_p_sum += rouge['rouge-1']['p']
    rouge_1_f_sum += rouge['rouge-1']['f']
    rouge_2_r_sum += rouge['rouge-2']['r']
    rouge_2_p_sum += rouge['rouge-2']['p']
    rouge_2_f_sum += rouge['rouge-2']['f']
    rouge_l_r_sum += rouge['rouge-l']['r']
    rouge_l_p_sum += rouge['rouge-l']['p']
    rouge_l_f_sum += rouge['rouge-l']['f']
    bleu_sum += bleu

def CalcStatG(rouge, bleu):
    global G_rouge_1_r_sum
    global G_rouge_1_p_sum
    global G_rouge_1_f_sum
    global G_rouge_2_r_sum
    global G_rouge_2_p_sum
    global G_rouge_2_f_sum
    global G_rouge_l_r_sum
    global G_rouge_l_p_sum
    global G_rouge_l_f_sum
    global G_bleu_sum
    G_rouge_1_r_sum += rouge['rouge-1']['r']
    G_rouge_1_p_sum += rouge['rouge-1']['p']
    G_rouge_1_f_sum += rouge['rouge-1']['f']
    G_rouge_2_r_sum += rouge['rouge-2']['r']
    G_rouge_2_p_sum += rouge['rouge-2']['p']
    G_rouge_2_f_sum += rouge['rouge-2']['f']
    G_rouge_l_r_sum += rouge['rouge-l']['r']
    G_rouge_l_p_sum += rouge['rouge-l']['p']
    G_rouge_l_f_sum += rouge['rouge-l']['f']
    G_bleu_sum += bleu
    


dataset_size = sys.argv[1]
if (int(dataset_size) == 0):
    dataset_size = "ALL"
    
test_num = int(sys.argv[2])
if (test_num == 0):
    test_num = 1350
    
rank = int(sys.argv[3])

is_gemma_train = sys.argv[4]

BASE_MODEL = "google/gemma-2b-it"
FINETUNE_MODEL = "./exper/MergeModel/summaryNEWS-gemma-2b-it-sum-ko.Rank" + str(rank) + "." + dataset_size


model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"":0})
finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)

pipe_finetuned = pipeline("text-generation", model=finetune_model, tokenizer=tokenizer, max_new_tokens=512)

# test_dataset = "dataset-500"
test_dataset = "dataset-SummaryNEWS-400"
dataset2 = load_from_disk(test_dataset)
doc = dataset2['validation']

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

rouge = Rouge()



for i in range(test_num):
    messages = [
        {
            "role": "user",
            "content": "다음 글을 요약해주세요:\n\n{}".format(doc['text'][i])
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
        Gemma_file_name = "./exper/gemma_out/Gemma_SummaryNEWS_output_" + str(i) + ".txt"
        Gemma_file = open(Gemma_file_name, "w")
        Gemma_file.write(Gemma_out)
        Gemma_file.close()
        Gemma_time_name = "./exper/gemma_out/Gemma_SummaryNEWS_time_" + str(i) + ".txt"
        Gemma_time = open(Gemma_time_name, "w")
        Gemma_time.write(str(G_sec))
        Gemma_time.close()
    else:
        Gemma_file_name = "./exper/gemma_out/Gemma_SummaryNEWS_output_" + str(i) + ".txt"
        Gemma_file = open(Gemma_file_name, "r")
        Gemma_out = Gemma_file.read()
        Gemma_file.close()
        
        Gemma_time_name = "./exper/gemma_out/Gemma_SummaryNEWS_time_" + str(i) + ".txt"
        Gemma_time = open(Gemma_time_name, "r")
        G_sec = float(Gemma_time.read())
        Gemma_time.close()
        G_time += G_sec 
    
    reference = doc['summary'][i]
    
    output_file = "./exper/InferenceResult/SummaryNEWS.Rank" + str(rank) + "." + dataset_size + "_" + str(i) + ".out"
    f = open(output_file, "w")
    f.write("Input_file :\n")
    f.write(prompt)
    f.write("\n\nAdapter Fine-tuned Output :\n")
    f.write(Fine_tuned_out)
    f.write("\nAdapter Fine-tuning ROUGE Score -- ")
    FT_rouge = rouge.get_scores([Fine_tuned_out], [reference], avg=True)
    f.write(json.dumps(FT_rouge))
    f.write("\nAdapter Fine-tuning BLEU Score -- ")
    FT_bleu = sentence_bleu([reference.split()], Fine_tuned_out.split(), weights=(1, 0, 0, 0))
    f.write(str(FT_bleu))
    CalcStatF(FT_rouge, FT_bleu)
    f.write("\n\nGemma Output :\n")
    f.write(Gemma_out)
    f.write("\nGemma Model ROUGE Score -- ")
    G_rouge = rouge.get_scores([Gemma_out], [reference], avg=True)
    f.write(json.dumps(G_rouge))
    f.write("\nGemma Model BLEU Score -- ")
    G_bleu = sentence_bleu([reference.split()], Gemma_out.split(), weights=(1, 0, 0, 0))
    f.write(str(G_bleu))
    CalcStatG(G_rouge, G_bleu)
    f.write("\n\nLabel summary :\n")
    f.write(reference)
    f.close()
    
    # print("OUTPUT ", i)
    # print(outputs[0]["generated_text"][len(prompt):])
    # print()

avg_FT_time = FT_time/test_num
avg_G_time = G_time/test_num
avg_FT_rouge_1_r = rouge_1_r_sum/test_num
avg_FT_rouge_1_p = rouge_1_p_sum/test_num
avg_FT_rouge_1_f = rouge_1_f_sum/test_num
avg_FT_rouge_2_r = rouge_2_r_sum/test_num
avg_FT_rouge_2_p = rouge_2_p_sum/test_num
avg_FT_rouge_2_f = rouge_2_f_sum/test_num
avg_FT_rouge_l_r = rouge_l_r_sum/test_num
avg_FT_rouge_l_p = rouge_l_p_sum/test_num
avg_FT_rouge_l_f = rouge_l_f_sum/test_num
avg_G_rouge_1_r = G_rouge_1_r_sum/test_num
avg_G_rouge_1_p = G_rouge_1_p_sum/test_num
avg_G_rouge_1_f = G_rouge_1_f_sum/test_num
avg_G_rouge_2_r = G_rouge_2_r_sum/test_num
avg_G_rouge_2_p = G_rouge_2_p_sum/test_num
avg_G_rouge_2_f = G_rouge_2_f_sum/test_num
avg_G_rouge_l_r = G_rouge_l_r_sum/test_num
avg_G_rouge_l_p = G_rouge_l_p_sum/test_num
avg_G_rouge_l_f = G_rouge_l_f_sum/test_num
avg_FT_bleu = bleu_sum/test_num
avg_G_bleu = G_bleu_sum/test_num

output_report = "./exper/InferenceReport/SummaryNEWS_Inference.Rank" + str(rank) + "." + dataset_size + ".out"
f2 = open(output_report, "w")
f2.write("Dataset: Summary Conversation\n")
f2.write("Test Dataset Size: " + str(test_num) + "\n")
f2.write("Fine-tuning Inference Time: " + str(avg_FT_time) + "\n")
f2.write("Gemma Model Inference Time: " + str(avg_G_time) + "\n")
f2.write("\nFine-tuning Score:\n")
f2.write("ROUGE-1 r: " + str(avg_FT_rouge_1_r) + "\n")
f2.write("ROUGE-1 p: " + str(avg_FT_rouge_1_p) + "\n")
f2.write("ROUGE-1 f: " + str(avg_FT_rouge_1_f) + "\n")
f2.write("ROUGE-2 r: " + str(avg_FT_rouge_2_r) + "\n")
f2.write("ROUGE-2 p: " + str(avg_FT_rouge_2_p) + "\n")
f2.write("ROUGE-2 f: " + str(avg_FT_rouge_2_f) + "\n")
f2.write("ROUGE-l r: " + str(avg_FT_rouge_l_r) + "\n")
f2.write("ROUGE-l p: " + str(avg_FT_rouge_l_p) + "\n")
f2.write("ROUGE-l f: " + str(avg_FT_rouge_l_f) + "\n")
f2.write("ROUGE-l f: " + str(avg_FT_rouge_l_f) + "\n")
f2.write("BLEU: " + str(avg_FT_bleu) + "\n")
f2.write("\nGemma Model Score\n")
f2.write("ROUGE-1 r: " + str(avg_G_rouge_1_r) + "\n")
f2.write("ROUGE-1 p: " + str(avg_G_rouge_1_p) + "\n")
f2.write("ROUGE-1 f: " + str(avg_G_rouge_1_f) + "\n")
f2.write("ROUGE-2 r: " + str(avg_G_rouge_2_r) + "\n")
f2.write("ROUGE-2 p: " + str(avg_G_rouge_2_p) + "\n")
f2.write("ROUGE-2 f: " + str(avg_G_rouge_2_f) + "\n")
f2.write("ROUGE-l r: " + str(avg_G_rouge_l_r) + "\n")
f2.write("ROUGE-l p: " + str(avg_G_rouge_l_p) + "\n")
f2.write("ROUGE-l f: " + str(avg_G_rouge_l_f) + "\n")
f2.write("ROUGE-l f: " + str(avg_G_rouge_l_f) + "\n")
f2.write("BLEU: " + str(avg_G_bleu) + "\n")
f2.close()
