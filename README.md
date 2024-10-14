# 安装

## 按照官方文档搭建环境
官方文档 (https://github.com/FunAudioLLM/CosyVoice)

- 克隆代码库
``` sh
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
```
如果由于网络故障导致克隆子模块失败，请运行以下命令直至成功。
``` sh
cd CosyVoice
git submodule update --init --recursive
```

- 安装 MiniConda
    
  下载：https://docs.conda.io/en/latest/miniconda.html   
  
  系统环境变量Path添加miniconda下的三个路径:
    
  D:\Program Files\Miniconda3  
  D:\Program Files\Miniconda3\Library\bin  
  D:\Program Files\Miniconda3\Scripts  
  
  打开 系统属性 > 高级系统设置 > 环境变量，可以设置存储 conda 环境的路径和 conda 包的路径。

  变量名：CONDA_ENVS_PATH
  变量值：conda 环境的路径，例如 D:\Program Files\Miniconda3\envs

  变量名：CONDA_PKGS_DIRS
  变量值：conda 包的路径，例如 D:\Program Files\Miniconda3\packages

    
- 创建虚拟环境
``` sh
conda create -n cosyvoice python=3.8
conda activate cosyvoice
```

- 安装依赖
pynini 是 WeTextProcessing 所必需的，使用 conda 安装。
``` sh
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

## 解决 sox 兼容性问题
``` sh
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

## 下载预训练模型
强烈建议下载以下预训练模型和资源：
- CosyVoice-300M
- CosyVoice-300M-SFT
- CosyVoice-300M-Instruct
- CosyVoice-ttsfrd
如果您是该领域的专家，并且只对从头开始训练自己的 CosyVoice 模型感兴趣，则可以跳过此步骤。

***SDK模型下载***
``` sh
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='pretrained_models/CosyVoice-300M-25Hz')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

***git模型下载，请确保已安装git lfs***
``` sh
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-300M-25Hz.git pretrained_models/CosyVoice-300M-25Hz
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git pretrained_models/CosyVoice-300M-Instruct
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

## 额外步骤（可选）
您可以解压缩 ttsfrd 资源并安装 ttsfrd 包以获得更好的文本规范化性能。请注意，这一步不是必需的。

``` sh
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl
```
