import os  # 导入os模块以便进行文件和目录操作
import sys  # 导入sys模块以便处理Python解释器的参数和操作
from generate_audio import generate_audio  # 导入新的生成音频的函数

# 设置路径和导入 CosyVoice 库
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的目录
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))  # 将第三方库路径添加到系统路径中

# --------------- 主函数 ---------------
def main():
    while True:
        tts_text = input("请输入合成文本（输入'exit'退出）：")
        if tts_text.lower() == 'exit':  # 检查用户是否输入'exit'
            print("退出程序。")
            break  # 退出循环
        
        generate_audio(tts_text)  # 生成并播放音频

if __name__ == '__main__':
    main()