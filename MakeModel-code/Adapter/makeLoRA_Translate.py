import torch
from datasets import Dataset, load_dataset, load_from_disk
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import sys
import math
import time
import datetime


torch.manual_seed(0)

def generate_prompt(example):
    global tans_type
    dest_lan = ''
    if tans_type[2:] == 'ko':
        dest_lan = '한국어'
    elif tans_type[2:] == 'en':
        dest_lan = '영어'
    elif tans_type[2:] == 'cn':
        dest_lan = '중국어'
    elif tans_type[2:] == 'jp':
        dest_lan = '일본어'
    
    prompt_list = []

    for i in range(len(example['src'])):
        if tans_type == 'kocn':
            prompt_list.append(r"""<bos><start_of_turn>user
请将以下文章翻译成中文:


{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(example['src'][i], example['dest'][i]))

        else:
            prompt_list.append(r"""<bos><start_of_turn>user
다음 글을 {}로 번역해주세요:
            
{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(dest_lan, example['src'][i], example['dest'][i]))
    return prompt_list


dataset_size = sys.argv[1]
if (int(dataset_size) == 0):
    dataset_size = "ALL"
    
rank = int(sys.argv[2])

dataset_type = sys.argv[3]

tans_type = sys.argv[4]

dataset2_path = 'datasetTranslate' + dataset_type + '-' + tans_type + '-' + str(int(int(dataset_size)/2))
dataset3_path = 'datasetTranslate' + str(int(dataset_type)+1) + '-' + tans_type + '-' + str(int(int(dataset_size)/2))

dataset2 = load_from_disk(dataset2_path)
dataset3 = load_from_disk(dataset3_path)

dataset = DatasetDict({
    'train': Dataset.from_pandas(pd.concat([dataset2['train'].to_pandas(), dataset3['train'].to_pandas()], ignore_index=True)),
    'validation': Dataset.from_pandas(pd.concat([dataset2['validation'].to_pandas(), dataset3['validation'].to_pandas()], ignore_index=True))
})

BASE_MODEL = "google/gemma-2b-it"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)
tokenizer.padding_side = 'right'
tokenizer.truncation = True 
tokenizer.model_max_length = 256  

train_data = dataset['train']



PARAMS = dict()
PARAMS['DATASET_size'] = dataset_size
PARAMS['LoRA_rank'] = rank
PARAMS['epoch'] = 20
PARAMS['batch_size'] = 20
PARAMS['learning_rate'] = 5e-4

lora_config = LoraConfig(
    r=PARAMS['LoRA_rank'],
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)
tokenizer.padding_side = 'right'

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    max_seq_length=256,
    args=TrainingArguments(
        output_dir="./exper/TrainingOutput_Translate",
        num_train_epochs = PARAMS['epoch'],
        #max_steps=75,
        per_device_train_batch_size=PARAMS['batch_size'],
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        warmup_steps=2,
        learning_rate=PARAMS['learning_rate'],
        fp16=True,
        logging_steps=1,
        report_to=["tensorboard"],  # TensorBoard 활성화
        logging_dir="./exper/TrainingOutput_log/Translate_" + tans_type + "_Log",  # 로그 디렉토리 지정
    ),
    peft_config=lora_config,
    formatting_func=generate_prompt,
)

print('---------------------------------------------------------------------------')
print('DATASET size: ' + dataset_size)
print('LoRA Config:')
print('    Rank: ', PARAMS['LoRA_rank'])
print('Traing parameter:')
print('    epoch: ', PARAMS['epoch'])
print('    batch size: ', PARAMS['batch_size'])
print('    leaning rate: ', PARAMS['learning_rate'])
print('---------------------------------------------------------------------------')

start = time.time()
trainer.train()
end = time.time()

ADAPTER_MODEL = "./exper/Adapter/LoRA_adapter-Translate" + dataset_type + "-" + tans_type + ".Rank" + str(rank) + "." + dataset_size

trainer.model.save_pretrained(ADAPTER_MODEL)

sec = (end - start)
result = datetime.timedelta(seconds=sec)
print("Training time: ", end='')
print(result)