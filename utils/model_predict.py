import json
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import ast

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 强制将torch设置为使用第0号GPU
torch.cuda.set_device(0)

csv_path = ''  # 请将此处替换为你的实际文件路径
data = pd.read_csv(csv_path, encoding='utf-8')
print(f'unlabel总条数为{len(data)}')
data = data[:]
print(f'本次解决的unlabel数量为{len(data)}')

def predict(messages, model, tokenizer):
    device = "cuda"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)  # 添加 padding=True

    with torch.no_grad():  # 禁用梯度计算
        generated_ids = model.generate(model_inputs.input_ids, 
                                       attention_mask=model_inputs.attention_mask,  # 显式传递 attention_mask
                                       max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def clean_and_merge_response(response):
    # 去除多余的换行符
    response = response.replace('\n', '').replace('\r', '')

    # 检查并修复错误的拼接
    if response.startswith('[') and response.endswith(']'):
        try:
            # 尝试解析为单个列表
            parsed_response = ast.literal_eval(response)
            if isinstance(parsed_response, list):
                return parsed_response
        except (ValueError, SyntaxError):
            # 如果解析失败，尝试手动修复
            try:
                # 查找 '][' 的位置并将其替换为 '], ['
                # 然后合并所有列表
                fixed_response = response.replace('][', '], [')
                parsed_response = ast.literal_eval(fixed_response)

                # 检查是否为多层嵌套列表
                if all(isinstance(i, list) for i in parsed_response):
                    merged_response = []
                    for sublist in parsed_response:
                        merged_response.extend(sublist)
                    return merged_response
            except (ValueError, SyntaxError):
                # 如果仍然无法解析，则返回原始响应以供进一步检查
                return response
    return response

viewpoints = [
    #候选观点
]

model_dir = ""#模型base
lora_dir = ""#微调权重

# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)

# 加载训练好的Lora模型
model = PeftModel.from_pretrained(model, model_id=lora_dir)

# 将模型设置为评估模式
model.eval()

# 初始化列表来存储JSON数据
json_list = []

# 逐行处理数据并将结果添加到列表中


before_row = None
for index, row in tqdm(data.iterrows(), total=len(data)):
    try:
        #------------------推理---------------------------
        input_text = row['text_raw']  # 修改为当前行的文本
        test_texts = {
            "instruction": "现在有一段原文，请你找出匹配这段原文的观点，并输出其对应情感，输出格式如下：1.平台应公开封禁原因，尊重用户知情权。-情感。输出格式如下：1.平台应公开封禁原因，尊重用户知情权。-情感。"
                          "可多选可多选,要输出观点和情感，要输出观点和情感,请不要忘记输出情感，请不要忘记输出情感，情感是针对该观点的支持、中立或反对，",
            "viewpoint": f'观点如下，不要选择以下之外的观点{viewpoints}',
            "input": f"文本:{input_text}"
        }
        instruction = test_texts['instruction']
        viewpoint = test_texts["viewpoint"]
        input_value = test_texts['input']

        if before_row is None or input_text != before_row:
            messages_1 = [
                {"role": "system", "content": f"{instruction},{viewpoint},输出样例如下{example1},{example2}，{example3}"},
                {"role": "user", "content": f"{input_value}"}
            ]

            response = predict(messages_1, model, tokenizer)  # 推理结果喂response
            response = response.strip()
            print(response)
            response = clean_and_merge_response(response)  # 清理并合并可能的多重列表

        message = {
            "user_id": row['user_id'],
            "text_raw": row['text_raw'],
            "screen_name": row['screen_name'],
            "event_id": row['event_id'],
            "url": row['url'],
            "result": response
        }
        json_list.append(message)
        before_row = input_text  # 暂存上一组的文本

    except Exception as e:
        print(f"处理第 {index} 行时发生错误: {e}")
        # 可以选择将错误信息附加到json_list中，或者进行其他错误处理
        message = {
            "user_id": row['user_id'],
            "text_raw": row['text_raw'],
            "error": str(e)
        }
        json_list.append(message)
        continue  # 继续处理下一行

index = '25001_35000'
json_path = f''
# 将数据保存到JSON文件
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_list, json_file, ensure_ascii=False, indent=4)

print(f'JSON文件已保存到: {json_path}')
