"""先 patch JSON 的 numpy 类型处理，再启动 verl 的 Hydra 入口。

verl 会用 json 写 rollout / validation 记录。部分指标来自 numpy，
类型可能是 np.int64、np.float64、ndarray；标准库 json 默认不认识这些
类型。这个 wrapper 会先给当前 Python 进程安装一个兜底转换逻辑，再
import 并运行 verl。
"""

import json
import sys

import numpy as np

# 保留标准库 json 原本的 default 逻辑。遇到我们不关心的类型时，
# 仍然交回原实现处理，避免改变其它 JSON 行为。
_original_default = json.JSONEncoder.default


def _numpy_safe_default(self, obj):
    # numpy 标量看起来像普通 int/float，但类型不是 Python 原生类型，
    # json.dumps/json.dump 默认会报 "Object of type int64 is not JSON serializable"。
    # 这里把它们转成 JSON 原生支持的 Python 类型。
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return _original_default(self, obj)


# 替换当前进程里的 JSONEncoder.default。之后 verl 内部任何 json.dump(s)
# 遇到 numpy 类型，都会自动走上面的转换逻辑。
json.JSONEncoder.default = _numpy_safe_default

# main_ppo.main 是 Hydra 入口。--config-path、--config-name 和配置覆盖项
# 会由 Hydra 直接从 sys.argv 读取，所以这个 wrapper 不解析、不改动参数，
# 只负责提前打补丁，然后调用 main()。
from verl.trainer.main_ppo import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
