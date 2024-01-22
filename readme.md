# VTuberBowTie
This repository contains the PyTorch implementation of the paper titled "Conan’s Bow Tie: A Streaming Voice Conversion for Real-Time VTuber Livestreaming."


# Installation
1. Clone this repo to your local machine:
```bash
git clone git@github.com:zju-muslab/VTuberBowTie.git
``` 
2. Create a virtual python environment with `python>=3.10`, `Pytorch>=2.1.0`, `cuda>=11.8` and install the required dependencies: 
```bash
# Create a new environment with pytorch
conda create -n vtuerbowtie python==3.10
conda activate vtuberbowte
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

3. Download data and model checkpoints from [google drive](https://drive.google.com/drive/folders/1udlOjOLfOz1sOuUsXJB4dxTSxYbkGanc?usp=sharing), then extract the corresponding files based on the specified file paths within the compressed package..

4. Download Virtual Audio Interafce such as [VAC](https://vac.muzychenko.net/en/) for windows or [SoundFlower](https://github.com/mattingalls/Soundflower) for Mac. This is necessary to route audio to other applications.

# Usage
For the convenience of configuration, we only provide the CLI usage. Run the main.py file for real-time voice conversion:
```python
python main.py
```


# Citation
If you use this code in your research, cite via the following BibTeX:
```
@inproceedings{vtuberbowtie,
    author = {Qianniu Chen, Zhehan Gu, Li Lu, Xiangyu Xu, Zhongjie Ba, Feng Lin, Zhengguang Liu and Kui Ren},
    title  = {{Conan’s Bow Tie}: A Streaming Voice Conversion for Real-Time VTuber
Livestreaming},
    year   = {2024}
}
```