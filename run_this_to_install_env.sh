source ~/.bashrc
conda create -y -n "clprgbd" python==3.10
conda activate clprgbd
conda install -y nvidia/label/cuda-12.8.0::cuda-toolkit
conda install nvitop
python -m pip install torch torchvision
python -m pip install git+https://github.com/huggingface/transformers  # Latest version for Qwen3-VL support
python -m pip install qwen-vl-utils==0.0.14  # For vision processing utilities
python -m pip install pillow