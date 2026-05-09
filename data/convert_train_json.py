import json

# 输入文件路径（你的 Alpaca 格式 JSON 数组文件）
input_file = "train.json"      # 请替换为实际文件名
# 输出文件路径（messages 格式的 JSONL）
output_file = "train_data.jsonl"

# 读取原始数据（假设是 JSON 数组）
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)   # 如果是 JSON 数组，这步能正确解析

# 逐条转换并写入 JSONL
with open(output_file, "w", encoding="utf-8") as f_out:
    for item in data:
        # 构建 user 的 content：将 instruction 和 input 拼接起来
        user_content = f"{item['instruction']}\n{item['input']}"
        assistant_content = item['output']
        
        # 按 messages 格式组织
        messages = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        # 写入一行
        f_out.write(json.dumps(messages, ensure_ascii=False) + "\n")

print(f"转换完成！已生成 {output_file}，共 {len(data)} 条数据。")