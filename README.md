# Agent4DevoSlimeLab

```bash
conda create -n slime python=3.13
conda activate slime
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install ultralytics --no-deps
pip install -U "triton-windows<3.5"
```

```pwsh
# 写入临时环境变量
$env:DASHSCOPE_API_KEY = "your_api_key_here"
```

## 训练