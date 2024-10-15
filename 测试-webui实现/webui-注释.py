import os  # 导入用于操作系统路径和文件的模块
import sys  # 导入用于访问 Python 解释器的变量和函数的模块
import argparse  # 导入用于解析命令行参数的模块
import gradio as gr  # 导入 Gradio 库，用于快速创建用户界面
import numpy as np  # 导入 NumPy 库，用于科学计算
import torch  # 导入 PyTorch 库，用于深度学习
import torchaudio  # 导入 Torchaudio 库，用于音频处理
import random  # 导入随机数生成库
import librosa  # 导入 Librosa 库，用于音频分析和处理

# 获取当前脚本所在目录的路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 将第三方库的路径添加到系统路径中，以便后续导入
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

# 从 CosyVoice 库导入所需的模块
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

# 定义推理模式列表
inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
# 定义不同推理模式下的操作步骤说明
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
# 定义流式推理的选项
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8   # 定义音频最大值阈值

# 生成随机种子的函数
def generate_seed():
    seed = random.randint(1, 100000000) # 生成一个1到1亿之间的随机整数作为种子
    return {
        "__type__": "update",
        "value": seed   # 返回种子值
    }

# 音频后处理函数
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    # 使用 Librosa 的 trim 函数去除音频静音部分
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    # 在音频后面追加一些静音以便延长音频长度
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech   # 返回处理后的音频

# 根据选择的推理模式返回对应的操作说明
def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

# 音频生成函数
def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    # 处理上传或录制的音频
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    # 检查自然语言控制模式的条件    if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if cosyvoice.frontend.instruct is False:    # 如果当前模型不支持指令模式
            gr.Warning('您正在使用自然语言控制模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M-Instruct模型'.format(args.model_dir))
            yield (target_sr, default_data)
        if instruct_text == '': # 如果未输入指令文本
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            yield (target_sr, default_data)
        if prompt_wav is not None or prompt_text != '': # 如果同时提供了音频和文本
            gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')

    # 检查跨语种复刻模式的条件    if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if cosyvoice.frontend.instruct is True: # 如果当前模型支持指令模式
            gr.Warning('您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M模型'.format(args.model_dir))
            yield (target_sr, default_data)
        if instruct_text != '': # 如果输入了指令文本
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None:  # 如果没有提供音频文件
            gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
            yield (target_sr, default_data)
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')

    # # 检查3s极速复刻和跨语种复刻模式的条件    if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:  # 如果没有提供音频文件
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            yield (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr: # 如果音频采样率低于要求
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (target_sr, default_data)

    # 检查预训练音色模式的条件  sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')

    # 检查3s极速复刻模式的条件  zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            yield (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    # 根据选定的模式进行不同的推理处理
    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')   # 记录日志
        set_all_random_seed(seed)   # 设置随机种子
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):   # 调用预训练音色的推理函数
            yield (target_sr, i['tts_speech'].numpy().flatten())    # 返回生成的音频数据
    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request') # 记录日志
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))    # 加载和处理提示音频
        set_all_random_seed(seed)   # 设置随机种子
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):   # 调用极速复刻的推理函数
            yield (target_sr, i['tts_speech'].numpy().flatten())    # 返回生成的音频数据
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request') # 记录日志
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))    # 加载和处理提示音频
        set_all_random_seed(seed)   # 设置随机种子
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):    # 调用跨语种复刻的推理函数
            yield (target_sr, i['tts_speech'].numpy().flatten())    # 返回生成的音频数据
    else:   # 指令模式的推理处理
        logging.info('get instruct inference request')  # 记录日志
        set_all_random_seed(seed)   # 设置随机种子
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):   # 调用指令的推理函数
            yield (target_sr, i['tts_speech'].numpy().flatten())    # 返回生成的音频数据


def main():
    # 使用 Gradio 创建用户界面
    with gr.Blocks() as demo:
        # 显示项目的描述和模型链接
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    预训练模型 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        # 输入合成文本的文本框
        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")
        with gr.Row():  # 创建一行布局
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0]) # 选择推理模式的单选框
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)    # 显示当前模式的操作步骤
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25)   # 选择音色的下拉框
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1]) # 选择流式推理的单选框
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)  # 设置速度的数值输入框
            with gr.Column(scale=0.25): # 创建一列布局
                seed_button = gr.Button(value="\U0001F3B2") # 用于生成随机种子的按钮
                seed = gr.Number(value=0, label="随机推理种子") # 显示当前随机种子的数值输入框

        with gr.Row():  # 创建一行布局
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')  # 上传音频文件的输入框
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件') # 录制音频文件的输入框
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')  # 输入提示文本的文本框
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')  # 输入指令文本的文本框

        generate_button = gr.Button("生成音频") # 生成音频的按钮

        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)    # 显示生成音频的输出框

        # 按钮的点击事件
        seed_button.click(generate_seed, inputs=[], outputs=seed)   # 生成随机种子
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed], # 点击生成按钮时触发音频生成
                              outputs=[audio_output])   # 输出生成的音频
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text]) # 根据选择的模式改变操作说明
        
    demo.queue(max_size=4, default_concurrency_limit=2) # 设置 Gradio 队列的最大大小和并发限制
    demo.launch(server_name='0.0.0.0', server_port=args.port)   # 启动 Gradio 应用


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--port',
                        type=int,
                        default=8000)   # 设置服务器端口，默认为8000
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='local path or modelscope repo id')    # 设置模型路径
    args = parser.parse_args()  # 解析命令行参数
    cosyvoice = CosyVoice(args.model_dir)  # 实例化 CosyVoice 模型
    sft_spk = cosyvoice.list_avaliable_spks()  # 获取可用的音色列表
    prompt_sr, target_sr = 16000, 22050  # 设置提示音频和目标音频的采样率
    default_data = np.zeros(target_sr)  # 创建默认音频数据
    main()
