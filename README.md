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
