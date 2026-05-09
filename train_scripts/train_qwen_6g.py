import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ================== 配置 ==================
model_name_or_path = r"D:\00_demo\pythonProject\my_models\Qwen2.5-0.5B-Instruct"   # 本地路径
dataset_path = "D:\\00_demo\\pythonProject\\train_data.jsonl"                     # 数据集路径
output_dir = "D:\\00_demo\\pythonProject\\outputs\\qwen_lora_6g"

# ================== 1. 4-bit 量化加载 ==================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)

# 准备模型进行 k-bit 训练
model = prepare_model_for_kbit_training(model)

# 开启梯度检查点
model.gradient_checkpointing_enable()

# ================== 2. LoRA 配置 ==================
lora_config = LoraConfig(
    r=4,                           # 降低 rank 节省显存
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================== 3. 数据集 ==================
dataset = load_dataset('json', data_files=dataset_path, split='train')

def preprocess(example):
    if "messages" in example:
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    else:
        # 根据你的实际格式调整
        text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
    return tokenizer(text, truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# ================== 4. 训练参数 ==================
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    bf16=True,                     # RTX3060 支持 bf16，若不可用则改 fp16=True
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",      # 8-bit 优化器
    report_to="tensorboard",
    remove_unused_columns=False,
)

# ====== 定义 data_collator ======
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    #tokenizer=tokenizer,
    data_collator=data_collator, 
    #data_collator=lambda data: {
    #    'input_ids': torch.stack([d['input_ids'] for d in data]),
    #    'attention_mask': torch.stack([d['attention_mask'] for d in data]),
    #    'labels': torch.stack([d['input_ids'] for d in data])
    #}
)

# ================== 5. 开始训练 ==================
trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)