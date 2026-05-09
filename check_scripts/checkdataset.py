import json

input_file = "your_dataset.json"   # 原数组文件
output_file = "train_data.jsonl"

with open(input_file, "r", encoding="utf-8") as fin:
    data = json.load(fin)   # 读取整个数组

with open(output_file, "w", encoding="utf-8") as fout:
    for item in data:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")