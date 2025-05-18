import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
import sys
import argparse
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = "cuda" if torch.cuda.is_available() else "cpu"
# 数据集转换函数
def csv_to_jsonl(csv_path, jsonl_path, model_type="glm4"):
    """
    将CSV数据集转换为大模型微调所需的JSONL格式
    
    参数:
        csv_path: CSV文件路径
        jsonl_path: 输出的JSONL文件路径
        model_type: 模型类型，可选值为 'glm4', 'medical', 'legal'
    """
    df = pd.read_csv(csv_path)
    messages = []
    
    # 根据不同模型类型设置不同的指令前缀
    instruction_prefix = {
        "glm4": "你是一个智能助手，请根据输入提供准确的回答",
        "medical": "你是一个医疗领域的专业助手，请根据医学知识回答以下问题",
        "legal": "你是一个法律领域的专业助手，请根据法律知识回答以下问题"
    }
    
    prefix = instruction_prefix.get(model_type, instruction_prefix["glm4"])
    
    # 逐行读取CSV文件
    for index, row in df.iterrows():
        # 确保CSV包含text和result列
        if 'text' not in row or 'result' not in row:
            continue
            
        text = row['text']
        result = row['result']
        
        message = {
            "instruction": prefix,
            "input": text,
            "output": result,
        }
        messages.append(message)
    
    # 保存为JSONL文件
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

# 数据预处理函数
def process_func(example, model_type="glm4"):
    """
    将数据集进行预处理
    
    参数:
        example: 数据样本
        model_type: 模型类型，可选值为 'glm4', 'medical', 'legal'
    """
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    
    # 根据不同模型类型设置不同的提示词
    prompt_templates = {
        "glm4": f"\n{example['instruction']}\n\n{example['input']}\n\n",
        "medical": f"\n{example['instruction']}\n\n医学问题: {example['input']}\n\n",
        "legal": f"\n{example['instruction']}\n\n法律问题: {example['input']}\n\n"
    }
    
    prompt = prompt_templates.get(model_type, prompt_templates["glm4"])
    
    instruction = tokenizer(prompt, add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 预测函数
def predict(messages, model, tokenizer):
    device = "cuda:0"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(response)
     
    return response
    
def train_model(csv_path, username, model_type="glm4"):
    """
    根据CSV数据训练模型
    
    参数:
        csv_path: CSV文件路径
        username: 用户名，用于创建用户专属的checkpoint目录
        model_type: 模型类型，可选值为 'glm4', 'medical', 'legal'
    
    返回:
        checkpoint_dir: 训练后的模型权重保存路径
    """
    # 模型路径映射
    model_paths = {
        "glm4": "/mnt/前端/model/GLM-4-main/THUDM/glm-4-9b-chat",
        "medical": "/mnt/前端/model/GLM-4-main/THUDM/glm-4-9b-chat",  # 可替换为医疗专用模型路径
        "legal": "/mnt/前端/model/GLM-4-main/THUDM/glm-4-9b-chat"    # 可替换为法律专用模型路径
    }
    
    # 获取对应模型路径
    model_dir = model_paths.get(model_type, model_paths["glm4"])
    
    # 创建用户专属的数据和checkpoint目录
    user_data_dir = f"/mnt/前端/uploads/{username}/data"
    user_checkpoint_dir = f"/mnt/前端/uploads/{username}/checkpoints/{model_type}"
    
    os.makedirs(user_data_dir, exist_ok=True)
    os.makedirs(user_checkpoint_dir, exist_ok=True)
    
    # 转换CSV为JSONL格式
    train_jsonl_path = f"{user_data_dir}/train.jsonl"
    csv_to_jsonl(csv_path, train_jsonl_path, model_type)
    
    # Transformers加载模型权重
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    model.config.use_cache = False  # 禁用 use_cache 以兼容梯度检查点
    
    print(f"模型运行设备: {next(model.parameters()).device}")
    print(f"使用模型类型: {model_type}")
    print(f"用户: {username}")
    
    # 加载训练集
    train_df = pd.read_json(train_jsonl_path, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    
    # 使用闭包来传递model_type参数
    def process_with_model_type(example):
        return process_func(example, model_type)
    
    train_dataset = train_ds.map(process_with_model_type, remove_columns=train_ds.column_names)

    # 检查数据集样本所在的设备
    if len(train_dataset) > 0:
        sample_data = next(iter(train_dataset))
        sample_input_ids = torch.tensor(sample_data['input_ids']).to(device)
        print(f"数据集样本所在设备: {sample_input_ids.device}")
    else:
        print("警告: 训练数据集为空!")
        return None
    
    # LoRA配置
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "activation_func", "dense_4h_to_h"],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1,  # Dropout 比例
    )
    
    model = get_peft_model(model, config).to(device)
    
    # 再次检查模型所在的设备
    print(f"LoRA 模型运行设备: {next(model.parameters()).device}")
    
    # 训练参数配置
    args = TrainingArguments(
        output_dir=user_checkpoint_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        num_train_epochs=3,  # 减少训练轮次，加快训练速度
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        fp16=torch.cuda.is_available(),  # 仅在CUDA可用时启用混合精度训练
    )
    
    # SwanLab 回调配置
    swanlab_callback = SwanLabCallback(
        project=f"{model_type}-finetune",
        experiment_name=f"{username}-{model_type}",
        description=f"用户{username}的{model_type}模型微调",
        config={
            "model": model_dir,
            "dataset": train_jsonl_path,
            "user": username,
            "model_type": model_type
        },
    )
    
    # 训练器配置
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model(user_checkpoint_dir)
    print(f"模型已保存到: {user_checkpoint_dir}")
    
    return user_checkpoint_dir

def main():
    """
    主函数，用于命令行调用
    """
    parser = argparse.ArgumentParser(description="模型训练脚本")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV文件路径")
    parser.add_argument("--username", type=str, required=True, help="用户名")
    parser.add_argument("--model_type", type=str, default="glm4", choices=["glm4", "medical", "legal"], help="模型类型")
    
    args = parser.parse_args()
    
    # 调用训练函数
    checkpoint_dir = train_model(args.csv_path, args.username, args.model_type)
    
    if checkpoint_dir:
        print(f"训练完成，模型保存在: {checkpoint_dir}")
        return 0
    else:
        print("训练失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())