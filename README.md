# Agent4DevoSlimeLab

conda create -n slime python=3.13
conda activate slime
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install ultralytics --no-deps
pip install -U "triton-windows<3.5"

## 训练