# from recognize_speech import recognize_speech
# from text_to_speech import text_to_speech
# from play_audio import play_audio
from run_model import run_model
from run_open_command import end_word, load_web, load_app_paths, judge_command, handle_open_command
# —————————————————————————————— 主函数 ——————————————————————————————————
def main():
    try:
        messages = []
        while True:
            # message = recognize_speech() # 用户输入
            message = input("请输入:") # 用户输入
            messages.append({'role': 'user', 'content': message})

            # 判断退出指令
            if message.strip().lower() in end_word:
                # text_to_speech('好的。')
                # play_audio("output.wav")
                print("对话结束，程序即将退出。")
                break

            # 加载json进行是否打开网站或应用的判断
            webs = load_web()
            apps = load_app_paths()

            # 判断是否打开网站或应用
            if judge_command(message, webs, apps):
                handle_open_command(message, webs, apps)
            else:
                # 如果以上都不是，调用大模型回复
                model_message = run_model(messages)
                messages.append({'role': 'assistant', 'content': model_message})
                # text_to_speech(model_message)
                # play_audio("output.wav")
                print(model_message)
                # print(messages)

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
