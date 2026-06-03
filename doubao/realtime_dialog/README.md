# 豆包实时语音对话

基于火山引擎豆包端到端实时语音大模型的实时对话模块，支持麦克风实时对话、音频文件输入和纯文本交互。

## 环境要求

- Python 3.10+
- 需要安装 PortAudio（PyAudio 依赖）
  - Windows: 通常 PyAudio wheel 自带
  - macOS: `brew install portaudio`

## 安装

```bash
pip install -r doubao/realtime_dialog/requirements.txt
```

## 配置

在项目根目录 `.env` 中填入火山引擎控制台密钥：

```
DOUBAO_APP_ID=你的AppID
DOUBAO_ACCESS_KEY=你的AccessKey
DOUBAO_APP_KEY=你的AppKey
```

可在 `doubao_config.py` 中修改：
- `speaker`：发音人（默认云洲男声）
  - `zh_female_vv_jupiter_bigtts`：vv 女声
  - `zh_female_xiaohe_jupiter_bigtts`：xiaohe 女声
  - `zh_male_yunzhou_jupiter_bigtts`：云洲男声
  - `zh_male_xiaotian_jupiter_bigtts`：小天男声
- `bot_name`、`system_role`、`speaking_style`：角色设定

## 使用

### 麦克风实时对话（建议戴耳机防止回声）

```python
import asyncio
from doubao.realtime_dialog.audio_manager import DialogSession
from doubao.realtime_dialog.doubao_config import ws_connect_config

session = DialogSession(ws_config=ws_connect_config, mod="audio")
asyncio.run(session.start())
```

### 纯文本交互

```python
import asyncio
from doubao.realtime_dialog.audio_manager import DialogSession
from doubao.realtime_dialog.doubao_config import ws_connect_config

session = DialogSession(ws_config=ws_connect_config, mod="text", recv_timeout=120)
asyncio.run(session.start())
```

### 音频文件输入

```python
import asyncio
from doubao.realtime_dialog.audio_manager import DialogSession
from doubao.realtime_dialog.doubao_config import ws_connect_config

session = DialogSession(ws_config=ws_connect_config, audio_file_path="your_audio.wav")
asyncio.run(session.start())
```

## 计费

按 token 计费，不按连接时长。静默不产生费用，只有实际语音交互时才计费。详见[计费文档](https://www.volcengine.com/docs/6561/1359370?lang=zh)。
