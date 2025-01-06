# -*- coding: utf-8 -*
"""
计算指标 困惑度（PPL） 的实现
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载tokenizer和模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 定义计算困惑度的函数
def calculate_perplexity(text):
    # 将文本转换为模型输入格式
    input_ids = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        # 获取逻辑回归（logits）
        outputs = model(input_ids)
        logits = outputs.logits
        # 将logits转换为概率分布
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # 获取实际token的索引
        token_idx = input_ids.view(-1)[1:]  # 忽略第一个token，因为它通常是特殊token
        # 获取实际token的概率
        token_probs = probs.gather(dim=-1, index=token_idx.unsqueeze(-1)).squeeze()
        # 计算对数概率
        log_probs = torch.log(token_probs)
        # 计算平均对数概率
        avg_log_prob = torch.mean(log_probs)
        # 计算困惑度
        perplexity = torch.exp(avg_log_prob)
    return perplexity.item()

# 用户输入的文本
text = "自然语言处理是人工智能的重要分支。"
ppl = calculate_perplexity(text)
print(f"文本的困惑度: {ppl:.2f}")