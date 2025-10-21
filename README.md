# Agent4DevoSlimeLab

```powershell
conda create -n slime python=3.13
conda activate slime
conda install open-world-agents::gstreamer-bundle
pip install owa
pip install virtual-uv uv
```

## Getting Started

```bash
conda activate slime
ocap DevoSlime.mcap --window-name DevoSlime --no-record-audio --fps 30
ocap DevoSlime.mcap
```

```bash
owl mcap info ./DevoSlime.mcap
```

```bash
cd open-world-agents/projects/owa-mcap-viewer
$env:EXPORT_PATH = "C:/workspace/github/Agent4DevoSlimeLab/mcap"
vuv install
uvicorn owa_viewer:app --host 0.0.0.0 --port 7860 --reload
```

``bash
python 
```
