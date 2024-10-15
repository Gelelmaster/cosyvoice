import os
import random
import numpy as np
import torch
import scipy.io.wavfile
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.common import set_all_random_seed
from pydub import AudioSegment
import simpleaudio as sa

# 设置路径和导入 CosyVoice 库
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = 'pretrained_models/CosyVoice-300M-Instruct'  # 默认模型目录的路径
DEFAULT_VOICE_ID = '中文女'  # 默认音色ID

# 初始化全局变量和模型
def initialize_globals(model_dir):
    global cosyvoice  # 声明全局变量
    cosyvoice = CosyVoice(model_dir)  # 创建CosyVoice模型实例

# --------------- 音频后处理 ---------------
def postprocess(speech, max_val=0.8, target_sr=22050, top_db=60, hop_length=220, win_length=440):
    # 使用Librosa的trim函数去除音频前后的静音部分
    speech, _ = librosa.effects.trim(speech, top_db=top_db, frame_length=win_length, hop_length=hop_length)
    # 如果音频的绝对值最大值超过设定的最大值，则进行归一化处理
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    # 返回后处理的音频，最后加上1秒的静音
    return torch.cat([speech, torch.zeros(1, int(target_sr * 1))], dim=1)

# --------------- 执行推理过程并生成音频 ---------------
def inference(tts_text, seed, stream, speed):
    set_all_random_seed(seed)  # 设置随机种子以确保结果可复现
    audio_gen = cosyvoice.inference_sft(tts_text, DEFAULT_VOICE_ID, stream=stream, speed=speed)  # 生成音频
    # 从生成器中提取音频数据
    audio_data = []
    for audio in audio_gen:
        audio_data.append(audio['tts_speech'].numpy().flatten())
    return np.concatenate(audio_data)  # 将所有片段连接成一个完整的音频数组

# --------------- 播放生成的音频 ---------------
def play_audio(audio, target_sr=22050):
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

    wave_obj = sa.WaveObject.from_wave_file(temp_file)  # 从临时文件创建WaveObject
    play_obj = wave_obj.play()  # 播放音频
    play_obj.wait_done()  # 等待播放完成

    os.remove(temp_file)  # 播放完成后删除临时文件

# --------------- 根据输入文本生成音频并播放 ---------------
def generate_audio(tts_text, model_dir=MODEL_DIR):
    seed = random.randint(1, 100000000)  # 生成随机种子
    stream = True  # 假定为非流式推理
    speed = 1.0  # 默认速度

    initialize_globals(model_dir)  # 初始化模型
    audio = inference(tts_text, seed, stream, speed)  # 调用推理函数生成音频

    # 保存生成的音频
    output_file = "output.wav"
    scipy.io.wavfile.write(output_file, 22050, audio)  # 保存音频为wav格式
    play_audio(audio)  # 播放生成的音频