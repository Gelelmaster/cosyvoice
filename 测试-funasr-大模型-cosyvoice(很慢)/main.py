import os
import sys

# 设置路径和导入 CosyVoice 库
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的目录
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))  # 将第三方库路径添加到系统路径中

from run_model import run_model
from run_open_command import end_word, load_web, load_app_paths, judge_command, handle_open_command

# 生成语音
from generate_audio import generate_audio

# 语音识别转文字
from funasr_recognize import record_audio
from funasr_recognize import transcribe_audio
# —————————————————————————————— 主函数 ——————————————————————————————————
def main():
    try:
        webs, apps = load_web(), load_app_paths() # 加载网站和应用路径
        while True:
            audio_buffer = record_audio()  # 录制音频
            print(audio_buffer)
            if audio_buffer is not None:
                message = transcribe_audio(audio_buffer)  # 识别音频
                print(message)

                # 判断退出指令
                if message.strip().lower() in end_word:
                    # text_to_speech('好的。')
                    # play_audio("output.wav")
                    print("对话结束，程序即将退出。")
                    break

                # 判断是否打开网站或应用
                if judge_command(message, webs, apps):
                    handle_open_command(message, webs, apps)
                
                # 如果以上都不是，调用大模型回复
                else:
                    model_message = run_model(message)
                    generate_audio(model_message)
                    print(model_message)
            else:
                continue

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()