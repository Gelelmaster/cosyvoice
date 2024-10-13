import os  # 导入os模块以便进行文件和目录操作
import sys  # 导入sys模块以便处理Python解释器的参数和操作
import random  # 导入random库以生成随机数
from cosyvoice.utils.common import set_all_random_seed  # 导入设置随机种子的工具函数
from audio_generator import generate_audio  # 导入新的生成音频的函数

# 设置路径和导入 CosyVoice 库
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的目录
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))  # 将第三方库路径添加到系统路径中

# --------------- 主函数 ---------------
def main():
    while True:  # 使用无限循环，允许用户多次输入文本
        tts_text = input("请输入合成文本（输入'exit'退出）：")  # 输入要合成的文本
        if tts_text.lower() == 'exit':  # 检查用户是否输入'exit'
            print("退出程序。")  # 提示退出
            break  # 退出循环
        
        generate_audio(tts_text)  # 生成并播放音频

if __name__ == '__main__':
    main()  # 调用主函数