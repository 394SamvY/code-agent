"""先 patch JSON 的 numpy 类型处理，再启动 verl 的 Hydra 入口。

verl 会用 json 写 rollout / validation 记录。部分指标来自 numpy，
类型可能是 np.int64、np.float64、ndarray；标准库 json 默认不认识这些
类型。这个 wrapper 会先给当前 Python 进程安装一个兜底转换逻辑，再
import 并运行 verl。
"""

import json
import sys

import hydra
import numpy as np
import ray

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


# 替换当前 driver 进程里的 JSONEncoder.default。validation 所在的
# Ray TaskRunner actor 会在 CodeAgentTaskRunner.run 内安装运行时补丁。
json.JSONEncoder.default = _numpy_safe_default

from verl.experimental.reward_loop import migrate_legacy_reward_impl  # noqa: E402
from verl.trainer.main_ppo import TaskRunner, run_ppo  # noqa: E402
from verl.utils.device import auto_set_device  # noqa: E402


class CodeAgentTaskRunner(TaskRunner):
    """TaskRunner that installs code-agent verl patches inside the Ray actor."""

    def run(self, config):
        from src.verl_runtime_patch import apply_patches

        apply_patches()
        return super().run(config)


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    task_runner_class = ray.remote(num_cpus=1)(CodeAgentTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)

if __name__ == "__main__":
    sys.exit(main())
