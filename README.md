# Qwen2.5-0.5B 测试用例通过判断微调

基于 Qwen2.5-0.5B-Instruct 模型，通过 LoRA + QLoRA 微调，实现根据测试用例执行记录自动判断是否通过（Pass/Fail）。

## 硬件要求
- GPU: NVIDIA RTX 3060 6GB（或同等显存）
- 系统: Windows 10/11

## 环境配置

```bash
conda create -n qwen_finetune python=3.10 -y
conda activate qwen_finetune
pip install -r requirements.txt

数据准备
原始数据格式参考 data/raw/ 下的 Excel 文件，包含 Measurements（测量文件名）和 Comments（备注）。

执行转换脚本生成 JSONL 训练数据：

bash
python data/convert_excel_to_jsonl.py
训练
bash
python scripts/train_qwen_6g.py
训练完成后，LoRA 权重保存在 outputs/qwen_lora_6g/。

测试
bash
python scripts/test_model.py
示例输入输出见 examples/。

模型导出（可选）
bash
python scripts/merge_lora.py
合并后的模型可用于常规推理部署。
