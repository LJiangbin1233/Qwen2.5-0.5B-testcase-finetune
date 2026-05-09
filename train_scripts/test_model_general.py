import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 基础模型路径（你下载的原始模型）
base_model_path = r"D:\my_models\Qwen2.5-0.5B-Instruct"
# LoRA 权重路径
lora_path = r"outputs/qwen_lora_6g"

# 加载 tokenizer 和基础模型
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, lora_path)

# 测试输入
prompt = ""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("模型回答：", response[len(prompt):])  # 只打印生成的部分