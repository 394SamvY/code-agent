"""先 patch JSON 的 numpy 类型处理，再启动 verl 的 Hydra 入口。

verl 会用 json 写 rollout / validation 记录。部分指标来自 numpy，
类型可能是 np.int64、np.float64、ndarray；标准库 json 默认不认识这些
类型。这个 wrapper 会先给当前 Python 进程安装一个兜底转换逻辑，再
import 并运行 verl。

调用链：
  bash evaluate_baseline_with_verl.sh
    → python3 scripts/verl_main_wrapper.py
      → main() [driver 进程]
        → json.JSONEncoder.default 替换为 _numpy_safe_default（driver 侧安全）
        → run_ppo(config, task_runner_class=CodeAgentTaskRunner)
          → verl 内部创建 Ray CPU actor，在 actor 里调用 CodeAgentTaskRunner.run(config)
            → apply_patches()  ← 在 Ray actor 内安装 patch，不干扰 GPU worker
            → super().run(config)  ← TaskRunner.run()，初始化 trainer/worker/agent loop

为什么必须通过 TaskRunner 子类安装 patch：
  - 如果在 sitecustomize.py 或 driver 进程全局 import 时安装 patch，
    Ray GPU worker 在 CUDA_VISIBLE_DEVICES 最终确定前就会 import verl/torch，
    导致 NCCL 报 Duplicate GPU detected
  - TaskRunner.run() 跑在 CPU Ray actor 里，此时 GPU actor 还未创建，
    在这里 import verl/torch 不会干扰 GPU worker 的 CUDA 绑定
  - 因此 patch 的安装点是"Ray CPU actor 内部"，而不是"Python 启动时"
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
# driver 进程也需要这个 patch，因为 driver 自己也会做 json.dumps 写配置等操作。
json.JSONEncoder.default = _numpy_safe_default

from verl.experimental.reward_loop import migrate_legacy_reward_impl  # noqa: E402
from verl.trainer.main_ppo import TaskRunner, run_ppo  # noqa: E402
from verl.utils.device import auto_set_device  # noqa: E402


class CodeAgentTaskRunner(TaskRunner):
    """TaskRunner that installs code-agent verl patches inside the Ray actor.

    为什么需要这个子类：
      TaskRunner 是 verl 的入口类，由 run_ppo() 通过 ray.remote() 创建为
      Ray CPU actor，然后在 actor 内调用 TaskRunner.run(config)。
      GPU worker（FSDP actor、SGLang server）是在 TaskRunner.run() 执行期间
      才被创建的，所以在这里 apply_patches() 不会影响 GPU 的 CUDA 绑定。

    调用时机（在 Ray CPU actor 内）：
      CodeAgentTaskRunner.run(config)
        → apply_patches()           # 安装 numpy JSON + validation partial dump
        → super().run(config)       # verl 原生流程：init_workers, _validate, ...
    """

    def run(self, config):
        from src.verl_runtime_patch import apply_patches

        apply_patches()
        return super().run(config)


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Hydra 入口，由 evaluate_baseline_with_verl.sh 通过 python3 直接调用。

    这个函数跑在 driver 进程（非 Ray actor），负责：
      1. auto_set_device: 设置 CUDA_VISIBLE_DEVICES 等设备相关环境变量
      2. migrate_legacy_reward_impl: 兼容 verl 旧版 reward 配置格式
      3. run_ppo: 创建 CodeAgentTaskRunner 的 Ray remote actor 并启动
    """
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    task_runner_class = ray.remote(num_cpus=1)(CodeAgentTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    sys.exit(main())
