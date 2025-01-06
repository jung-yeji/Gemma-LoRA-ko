import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import sys
import time
import datetime

dataset_size = sys.argv[1]
if (int(dataset_size) == 0):
    dataset_size = "ALL"


rank = int(sys.argv[2])

dataset_type = sys.argv[3]

tans_type = sys.argv[4]

BASE_MODEL = "google/gemma-2b-it"
# ADAPTER_MODEL = "./exper/Adapter/LoRA_adapter-Translate.Rank" + str(rank) + "." + dataset_size
ADAPTER_MODEL = "./exper/Adapter/LoRA_adapter-Translate" + dataset_type + "-" + tans_type + ".Rank" + str(rank) + "." + dataset_size




model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

start = time.time()
model = model.merge_and_unload()
end = time.time()

model.save_pretrained("./exper/MergeModel/Translate" + dataset_type + "-" + tans_type + "-gemma-2b-it-sum-ko.Rank" + str(rank) + "." + dataset_size)

sec = (end - start)
result = datetime.timedelta(seconds=sec)
output_file = "./exper/MergeReport/mergeTime_Translate" + dataset_type + "-" + tans_type + ".Rank" + str(rank) + "." + dataset_size + ".out"
f = open(output_file, "w")
f.write("DATASET size: " + dataset_size + "\n")
f.write("Rank: " + str(rank) + "\n")
f.write("Learning Rate: 2e-4\n")
f.write("Merge Time: " + str(result))
f.close()