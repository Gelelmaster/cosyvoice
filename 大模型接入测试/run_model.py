import os
import openai

# API Key（确保安全性，建议通过环境变量传入）
api_key = 'sk-60671a06987c4029b9ae5e31622e27a9'

def run_model(input_message):
    input_message = str(input_message)
    
    client = openai.OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 调用模型，确保 messages 参数格式正确
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": input_message}]  # 正确的参数格式
        )
        
        # 返回模型的回复
        return completion.choices[0].message.content
    
    except Exception as e:
        print(f"发生错误: {e}")  # 捕获并打印错误
