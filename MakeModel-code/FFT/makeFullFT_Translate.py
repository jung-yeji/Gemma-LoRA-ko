import torch
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import sys
import time
import datetime
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)

def generate_prompt_batched(examples):
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

    if tans_type == 'kocn':
        return {'text': [f"""<bos><start_of_turn>user
请将以下文章翻译成中文:

{passage}<end_of_turn>
<start_of_turn>model
{summary2}<end_of_turn><eos>""" for passage, summary2 in zip(examples['src'], examples['dest'])]}
        
    else:
        return {'text': [f"""<bos><start_of_turn>user
다음 글을 {dest_lan}로 번역해주세요:

{passage}<end_of_turn>
<start_of_turn>model
{summary2}<end_of_turn><eos>""" for passage, summary2 in zip(examples['src'], examples['dest'])]}




dataset_size = sys.argv[1]
if (int(dataset_size) == 0):
    dataset_size = "ALL"
    

dataset_type = sys.argv[2]

tans_type = sys.argv[3]

dataset2_path = 'datasetTranslate' + dataset_type + '-' + tans_type + '-' + str(int(int(dataset_size)/2))
dataset3_path = 'datasetTranslate' + str(int(dataset_type)+1) + '-' + tans_type + '-' + str(int(int(dataset_size)/2))

dataset2 = load_from_disk(dataset2_path)
dataset3 = load_from_disk(dataset3_path)

dataset = DatasetDict({
    'train': Dataset.from_pandas(pd.concat([dataset2['train'].to_pandas(), dataset3['train'].to_pandas()], ignore_index=True)),
    'validation': Dataset.from_pandas(pd.concat([dataset2['validation'].to_pandas(), dataset3['validation'].to_pandas()], ignore_index=True))
})

BASE_MODEL = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = 'right'
tokenizer.truncation = True
tokenizer.model_max_length = 256

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

train_data = dataset['train']

# num_proc 제거
train_data = train_data.map(generate_prompt_batched, batched=True)
train_data = train_data.map(
    lambda x: tokenizer(x['text'], truncation=True, padding='max_length'),
    batched=True
)
train_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])

PARAMS = {
    'epoch': 4,
    'batch_size': 20,  # 배치 사이즈 증가
    'learning_rate': 5e-6 # 학습률 증가
}

training_args = TrainingArguments(
    output_dir='./exper/TrainingoutFFT',
    num_train_epochs=PARAMS['epoch'],
    per_device_train_batch_size=PARAMS['batch_size'],
    learning_rate=PARAMS['learning_rate'],
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    prediction_loss_only=True,
    dataloader_num_workers=0,
    optim='adamw_torch',
    log_level='debug',
    gradient_accumulation_steps=1,  # 필요에 따라 조절
    warmup_steps=50,  # 워밍업 스텝 추가
    weight_decay=0.01,  # 가중치 감쇠 추가
    max_grad_norm=1.0,  # 그래디언트 클리핑
    report_to=["tensorboard"],  # TensorBoard 활성화
    logging_dir="./exper/TrainingOutput_log/Translate_" + tans_type + "_FFT_Log",  # 로그 디렉토리 지정
)

# DataCollatorForLanguageModeling을 사용하여 labels 생성
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 언어 모델링이 아닌 시퀀스 생성이므로 False로 설정
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print('---------------------------------------------------------------------------')
print('DATASET size: ' + dataset_size)
print('Traing parameter:')
print('    epoch: ', PARAMS['epoch'])
print('    batch size: ', PARAMS['batch_size'])
print('    leaning rate: ', PARAMS['learning_rate'])
print('---------------------------------------------------------------------------')

start = time.time()
trainer.train()
end = time.time()

save_directory = "./exper/FullFTModel/FFTModel-Translate." + tans_type + "." + dataset_size
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

sec = (end - start)
result = datetime.timedelta(seconds=sec)
print("Training time: ", end='')
print(result)