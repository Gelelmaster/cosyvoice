# ——————————————————————————————————— 字典 ——————————————————————————————————————————
# 结束对话关键字
end_word = ['结束','退下','退下吧','你退下吧','好的你退下吧']

# —————————————————————————————— open_functions.py ——————————————————————————————————
import os
import json
import re
# from text_to_speech import text_to_speech
# from play_audio import play_audio

# —————————————————————————————— 打开应用程序相关函数 —————————————————————————————————
import subprocess
# 定义存储应用程序路径的 JSON 文件
APP_PATHS_FILE = 'app_paths.json'

# 加载已知的应用程序路径
def load_app_paths():
    if os.path.exists(APP_PATHS_FILE):
        with open(APP_PATHS_FILE, 'r') as file:
            return json.load(file)  # 加载 JSON 文件并转换为字典
    return {}

# 保存新的应用程序路径
def save_app_paths(apps):
    with open(APP_PATHS_FILE, 'w') as file:
        json.dump(apps, file, indent=4)  # 将字典保存为 JSON 文件
    
# 根据命令打开对应的应用程序
def open_application(app_name, apps):
    subprocess.Popen(apps)
    # text_to_speech(f"已为你打开{app_name}")
    # play_audio("output.wav")

 # ———————————————————————————————— 打开网站相关函数 ——————————————————————————————————
import webbrowser
# 定义存储网站的 JSON 文件
WEB_URL = 'web_url.json'

# 加载已知的网站链接
def load_web():
    if os.path.exists(WEB_URL):
        with open(WEB_URL, 'r') as file:
            return json.load(file)  # 加载 JSON 文件并转换为字典
    return {}

# 保存新的网站链接
def save_web(webs):
    with open(WEB_URL, 'w') as file:
        json.dump(webs, file, indent=4)  # 将字典保存为 JSON 文件

def open_website(web_name , webs):
    webbrowser.open(webs)
    # text_to_speech(f"已为你打开{web_name}")
    # play_audio("output.wav")

 # ———————————————————————————————————————主函数——————————————————————————————————————
# 返回需要打开的网站或程序名，去掉前后空格转换为小写
def parse_command(message):
    if message.startswith("打开"):
        return message[2:].strip().lower()  # 去掉"打开"并去除空格
    else:
        return message.strip().lower()

# 判断用户输入是否是网站链接
def check_if_url(string):
    url_pattern = re.compile(
        r'^(https?:\/\/)?'  # 以 http:// 或 https:// 开头，整个部分是可选的
        r'(www\.)?'  # 以 www. 开头，这部分是可选的
        r'(([A-Za-z0-9]+[-._])*[A-Za-z0-9]+\.[A-Za-z]{2,6}(\.[A-Za-z]{2,6})?)'  # 域名部分，可以是主域名或者带有次级域名的形式
        r'(\:[0-9]{1,5})?'  # 端口号部分，可以有也可以没有，范围是1到5位数字
        r'(\/[-A-Za-z0-9._~:/?#[\]@!$&\'()*+,;%=]*)*'  # 路径部分，可以有多个层级
        r'(\?[-A-Za-z0-9._~:/?#[\]@!$&\'()*+,;%=]*)?'  # 查询字符串部分，以?开头，这部分是可选的
        r'(#[-A-Za-z0-9._~:/?#[\]@!$&\'()*+,;%=]*)?$'  # 锚点部分，以#开头，这部分是可选的
    )
    if url_pattern.match(string):
        return True
    return False

# 判断用户输入是否是系统路径
def check_if_file_path(string):
    return os.path.isfile(string)

# 验证用户输入
def validate_input(input_name, input_path):
    if not input_name:
        return False, "应用或网站名称不能为空"
    if not input_path:
        return False, "应用路径或URL不能为空"
    if check_if_url(input_path):
        return True, None
    if check_if_file_path(input_path):
        return True, None
    return False, "输入的路径或URL无效"

# 判断输入类型
def judge_command(message, webs, apps):
    command = parse_command(message)
    if message.strip().lower().startswith("打开"):
        return True
    if command in webs:
        return True
    if command in apps:
        return True
    return False

# 封装的处理打开网站或应用的函数
def handle_open_command(message, webs, apps):
    command = parse_command(message)

    if message.strip().lower() in {'打开'}:
        # text_to_speech("请问你要打开什么")
        # play_audio("output.wav")
        return

    if message.strip().lower() in {'打开网站'} :
        # text_to_speech('请问你要打开什么网站')
        # play_audio("output.wav")
        return

    # 网站命令
    if command in webs:
        open_website(command, webs[command])
        return

    # 应用程序命令
    if command in apps:
        open_application(command, apps[command])
        return

    # 不明命令，需要进一步判断
    # text_to_speech(f"抱歉，我还不会打开{command}，请输入{command}的链接或应用程序目录：")
    print(f"抱歉，我还不会打开{command}，请输入{command}的链接或应用程序目录：")
    # play_audio("output.wav")
    input_path = input(f"请输入{command}的链接或应用程序目录: ").lower()
    valid, error_message = validate_input(command, input_path)
    if valid:
        if check_if_url(input_path):
            webs[command] = input_path
            save_web(webs)
            print(f"{command} 已添加到已知网站列表，正在为你打开...")
            open_website(command, webs[command])
        else:
            apps[command] = input_path
            save_app_paths(apps)
            print(f"{command} 已添加到已知应用程序列表，正在为你打开...")
            open_application(command, apps[command])
    else:
        # text_to_speech(f"{error_message}")
        print(f"输入错误: {error_message}")
        # play_audio("output.wav")