import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import random
import librosa
import numpy as np
import scipy.io.wavfile
import tempfile  # 用于生成临时文件

# 设置路径和导入 CosyVoice 库
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

# 定义全局变量
inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8
target_sr = 22050  # 假定目标采样率
prompt_sr = 16000  # 假定提示音频采样率

# 初始化全局变量和模型
def initialize_globals(model_dir):
    global cosyvoice, sft_spk
    cosyvoice = CosyVoice(model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()

def generate_seed():
    """生成随机种子"""
    seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": seed}

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """音频后处理"""
    speech, _ = librosa.effects.trim(speech, top_db=top_db, frame_length=win_length, hop_length=hop_length)
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    return torch.cat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)

def change_instruction(mode):
    """根据选择的推理模式返回对应的操作说明"""
    return instruct_dict[mode]

def validate_inputs(mode, prompt_wav, instruct_text, prompt_text):
    """验证输入的有效性"""
    if mode == '自然语言控制' and instruct_text == '':
        return False, "请输入instruct文本"
    if mode == '跨语种复刻' and prompt_wav is None:
        return False, "请提供prompt音频"
    if mode == '3s极速复刻' and prompt_text == '':
        return False, "prompt文本为空，您是否忘记输入prompt文本？"
    return True, None

def inference(mode, tts_text, prompt_wav, instruct_text, sft_dropdown, seed, stream, speed):
    """执行推理过程并生成音频"""
    set_all_random_seed(seed)
    
    if mode == '预训练音色':
        for audio in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (audio['tts_speech'].numpy().flatten())  # 分段返回 numpy 数组格式音频
    elif mode in ['3s极速复刻', '跨语种复刻']:
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        inference_method = cosyvoice.inference_zero_shot if mode == '3s极速复刻' else cosyvoice.inference_cross_lingual
        for audio in inference_method(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (audio['tts_speech'].numpy().flatten())  # 分段返回 numpy 数组格式音频
    elif mode == '自然语言控制':
        for audio in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (audio['tts_speech'].numpy().flatten())  # 分段返回 numpy 数组格式音频



# 在代码开头定义操作说明字典
instruct_dict = {
    '预训练音色': '选择预训练音色并输入合成文本，点击生成音频。',
    '3s极速复刻': '上传prompt音频并输入prompt文本，点击生成音频。',
    '跨语种复刻': '上传prompt音频并输入prompt文本，点击生成音频。',
    '自然语言控制': '输入instruct文本，并选择对应的推理模式，点击生成音频。'
}

def create_ui(args):
    """创建 Gradio 用户界面"""
    with gr.Blocks() as demo:
        gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) ...")
        tts_text = gr.Textbox(label="输入合成文本", lines=1,
                              value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=1)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致...", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')

        generate_button = gr.Button("生成音频")
        audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)  # 自动播放关闭流式

        # 按钮的点击事件
        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])

    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)

def generate_audio(tts_text, mode, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    """生成音频并返回实时生成的音频块"""
    prompt_wav = prompt_wav_upload if prompt_wav_upload is not None else prompt_wav_record
    is_valid, warning_msg = validate_inputs(mode, prompt_wav, instruct_text, prompt_text)
    
    if not is_valid:
        yield gr.Warning(warning_msg)  # 返回警告信息
    
    # 推理并生成音频
    for audio in inference(mode, tts_text, prompt_wav, instruct_text, sft_dropdown, seed, stream, speed):
        if isinstance(audio, tuple):
            audio = audio[1]  # 如果返回的是元组，获取音频数据部分
        
        # 将生成的音频数据实时返回，而不是生成整个文件
        yield (22050, audio)  # 假定 target_sr = 22050
    

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice-300M',
                        help='local path or modelscope repo id')
    
    args = parser.parse_args()
    initialize_globals(args.model_dir)  # 初始化全局变量和模型
    global default_data  # 需要在函数中修改全局变量
    default_data = np.zeros(target_sr)  # 创建默认音频数据
    create_ui(args)  # 将 args 传递给 create_ui 函数

if __name__ == '__main__':
    main()  # 调用主函数
