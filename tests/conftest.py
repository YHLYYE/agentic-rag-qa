"""测试环境准备。

**为什么需要这个文件**：测试本身不依赖网络（模型权重已缓存在本地
`data/models/` 与 HuggingFace 缓存里），但 `transformers` / `huggingface_hub`
在实例化模型前会去探测 hub 元信息。网络正常时这只是几秒；**网络不通或很慢时
它会卡在重试超时上，让整个 `pytest` 看起来像死掉了。**

实测（同一台机器、同一份代码）：
    · 联网        160 passed in 42.65s
    · 显式离线    160 passed in 17.41s
    · 网络异常    卡住十分钟以上没有任何输出

所以这里把离线模式设为默认：本地有缓存就用缓存，缺失的模型会**立刻**报
「找不到本地文件」——比挂在网络超时里好诊断得多。

需要联网拉模型时可以覆盖：
    $env:HF_HUB_OFFLINE=0; python -m pytest      # PowerShell
    HF_HUB_OFFLINE=0 python -m pytest            # bash
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
