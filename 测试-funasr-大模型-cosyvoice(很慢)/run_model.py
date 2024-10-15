import os
import openai

# 从环境变量中获取 API Key，确保安全性
api_key = os.getenv('OPENAI_API_KEY')  # API Key 安全存储在环境变量中
if not api_key:
    raise ValueError("API Key 未设置，请在环境变量中配置 OPENAI_API_KEY")

def run_model(input_message):
    input_message = str(input_message)
    
    client = openai.OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": input_message}]  # 正确的参数格式
        )
        return completion.choices[0].message.content # 返回模型的回复
    
    except Exception as e:
        print(f"发生错误: {e}")  # 捕获并打印错误
