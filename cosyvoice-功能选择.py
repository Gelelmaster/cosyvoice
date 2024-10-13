import os  # 导入os模块以便进行文件和目录操作
import sys  # 导入sys模块以便处理Python解释器的参数和操作
import numpy as np  # 导入NumPy库用于数值计算
import torch  # 导入PyTorch库用于深度学习
import random  # 导入random库以生成随机数
import librosa  # 导入Librosa库用于音频分析
import scipy.io.wavfile  # 导入SciPy库以处理wav文件的读写
from cosyvoice.cli.cosyvoice import CosyVoice  # 从CosyVoice库导入CosyVoice类
from cosyvoice.utils.file_utils import load_wav  # 导入加载wav文件的工具函数
from cosyvoice.utils.common import set_all_random_seed  # 导入设置随机种子的工具函数
from pydub import AudioSegment  # 导入Pydub库用于处理音频段
import simpleaudio as sa  # 导入SimpleAudio库用于播放音频

# 设置路径和导入 CosyVoice 库
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的目录
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))  # 将第三方库路径添加到系统路径中

# 定义全局变量
max_val = 0.8  # 音频的最大值
target_sr = 22050  # 设定目标音频的采样率
prompt_sr = 16000  # 设定提示音频的采样率
MODEL_DIR = 'pretrained_models/CosyVoice-300M'  # 默认模型目录的路径

# 初始化全局变量和模型
def initialize_globals(model_dir):
    """初始化全局变量和模型"""
    global cosyvoice, sft_spk  # 声明全局变量
    cosyvoice = CosyVoice(model_dir)  # 创建CosyVoice模型实例
    sft_spk = cosyvoice.list_avaliable_spks()  # 列出可用的音色

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """音频后处理"""
    # 使用Librosa的trim函数去除音频前后的静音部分
    speech, _ = librosa.effects.trim(speech, top_db=top_db, frame_length=win_length, hop_length=hop_length)
    # 如果音频的绝对值最大值超过设定的最大值，则进行归一化处理
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    # 返回后处理的音频，最后加上0.2秒的静音
    return torch.cat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)

def validate_inputs(mode, prompt_wav, instruct_text, prompt_text):
    """验证输入的有效性"""
    # 检查模式和输入文本的有效性
    if mode == '自然语言控制' and instruct_text == '':
        return False, "请输入instruct文本"  # 如果是自然语言控制模式，但instruct文本为空
    if mode == '跨语种复刻' and prompt_wav is None:
        return False, "请提供prompt音频"  # 如果是跨语种复刻模式，但没有提供音频
    if mode == '3s极速复刻' and prompt_text == '':
        return False, "prompt文本为空，您是否忘记输入prompt文本？"  # 如果是3s极速复刻模式，但prompt文本为空
    return True, None  # 输入有效

def inference(mode, tts_text, prompt_wav, instruct_text, sft_dropdown, seed, stream, speed):
    """执行推理过程并生成音频"""
    set_all_random_seed(seed)  # 设置随机种子以确保结果可复现

    # 处理预训练音色模式
    if mode == '预训练音色':
        if sft_dropdown is None:
            raise ValueError("请选择有效的音色！")  # 确保音色ID不为None
        audio_gen = cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed)
        # 从生成器中提取音频数据
        audio_data = []
        for audio in audio_gen:
            audio_data.append(audio['tts_speech'].numpy().flatten())
        return np.concatenate(audio_data)  # 将所有片段连接成一个完整的音频数组

    # 处理3s极速复刻和跨语种复刻模式
    elif mode in ['3s极速复刻', '跨语种复刻']:
        # 处理提示音频
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        # 根据模式选择推理方法
        inference_method = cosyvoice.inference_zero_shot if mode == '3s极速复刻' else cosyvoice.inference_cross_lingual
        audio_gen = inference_method(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed)
        # 从生成器中提取音频数据
        audio_data = []
        for audio in audio_gen:
            audio_data.append(audio['tts_speech'].numpy().flatten())
        return np.concatenate(audio_data)  # 将所有片段连接成一个完整的音频数组

    # 处理自然语言控制模式
    elif mode == '自然语言控制':
        audio_gen = cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed)
        # 从生成器中提取音频数据
        audio_data = []
        for audio in audio_gen:
            audio_data.append(audio['tts_speech'].numpy().flatten())
        return np.concatenate(audio_data)  # 将所有片段连接成一个完整的音频数组

def play_audio(audio):
    """播放生成的音频"""
    # 将 numpy 数组转换为 pydub 音频段
    audio_segment = AudioSegment(
        audio.tobytes(),  # 将numpy数组转换为字节
        frame_rate=target_sr,  # 设置帧率为目标采样率
        sample_width=audio.dtype.itemsize,  # 设置样本宽度为数据类型的大小
        channels=1  # 设置音频通道为单声道
    )
    # 保存临时文件
    temp_file = "temp_output.wav"  # 临时音频文件名
    audio_segment.export(temp_file, format="wav")  # 导出音频为wav格式

    # 播放音频
    wave_obj = sa.WaveObject.from_wave_file(temp_file)  # 从临时文件创建WaveObject
    play_obj = wave_obj.play()  # 播放音频
    play_obj.wait_done()  # 等待播放完成

    # 删除临时文件
    os.remove(temp_file)  # 播放完成后删除临时文件

def main():
    """主函数"""
    initialize_globals(MODEL_DIR)  # 初始化全局变量和模型

    # 用户输入
    mode = input("请输入推理模式（预训练音色、3s极速复刻、跨语种复刻、自然语言控制）：")  # 输入推理模式
    if mode == '预训练音色':
        print("可用音色：", sft_spk)  # 输出可用的音色
        sft_dropdown = input("请选择音色ID：")  # 让用户选择音色ID
    else:
        sft_dropdown = None  # 其他模式下不需要音色ID

    tts_text = input("请输入合成文本：")  # 输入要合成的文本
    prompt_wav = None  # 根据需要可以添加文件上传的功能
    prompt_text = ""  # 提示文本初始化为空
    instruct_text = ""  # 指令文本初始化为空

    # 如果选择了需要prompt的模式
    if mode in ['3s极速复刻', '跨语种复刻']:
        prompt_wav = input("请输入prompt音频文件路径：")  # 假定用户提供路径
        prompt_text = input("请输入prompt文本：")  # 输入提示文本
    elif mode == '自然语言控制':
        instruct_text = input("请输入instruct文本：")  # 输入指令文本

    # 生成音频
    seed = random.randint(1, 100000000)  # 生成随机种子
    stream = False  # 假定为非流式推理
    speed = 1.0  # 默认速度

    audio = inference(mode, tts_text, prompt_wav, instruct_text, sft_dropdown, seed, stream, speed)  # 调用推理函数生成音频

    # 保存生成的音频
    output_file = "output.wav"  # 输出音频文件名
    scipy.io.wavfile.write(output_file, target_sr, audio)  # 保存音频为wav格式
    print(f"音频已保存为 {output_file}")  # 提示音频保存成功

    # 播放生成的音频
    play_audio(audio)  # 播放生成的音频

if __name__ == '__main__':
    main()  # 调用主函数
