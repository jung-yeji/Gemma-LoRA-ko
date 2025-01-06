import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import sys
import time
import datetime
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(0)

# cuDNN 최적화 활성화 (필요 시)
# torch.backends.cudnn.benchmark = True

def generate_prompt_batched(examples):
    return {'text': [f"""<bos><start_of_turn>user
다음 글을 요약해주세요:

{passage}<end_of_turn>
<start_of_turn>model
{summary2}<end_of_turn><eos>""" for passage, summary2 in zip(examples['passage'], examples['summary3'])]}

dataset_size = sys.argv[1]
if int(dataset_size) == 0:
    dataset_size = "ALL"

dataset_path = 'dataset-' + dataset_size
dataset = load_from_disk(dataset_path)

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
    logging_dir="./exper/TrainingOutput_log/Summary_FFT_Log",  # 로그 디렉토리 지정
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
print('DATASET size:', dataset_size)
print('Training parameters:')
print('    Epochs:', PARAMS['epoch'])
print('    Batch size:', PARAMS['batch_size'])
print('    Learning rate:', PARAMS['learning_rate'])
print('---------------------------------------------------------------------------')

start = time.time()
trainer.train()
end = time.time()

save_directory = "./exper/FullFTModel/FFTModel-Summary." + dataset_size
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

sec = end - start
result = datetime.timedelta(seconds=sec)
print("Training time:", result)
