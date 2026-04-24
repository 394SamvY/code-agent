"""
安全代码执行沙箱
================

使用 subprocess 在独立进程中执行 Python 代码，并提供：

- stdin 写入
- stdout/stderr 捕获
- timeout
- return code
- runtime 统计

它是环境的最底层执行层，只负责“怎么跑代码”，不负责 OJ verdict、测试循环、
submission limit 或 reward。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """代码执行结果。

    这是 sandbox 层返回给 judge 层的原始执行信息。上层会再把它解释为
    `accepted` / `wrong_answer` / `runtime_error` / `time_limit_exceeded`。
    """

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    runtime_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """Whether the subprocess exited normally without timeout."""
        return self.returncode == 0 and not self.timed_out


def execute_code(
    code: str,
    stdin: str = "",
    timeout: int | float = 5,
    max_output_chars: int = 4096,
) -> ExecutionResult:
    """在子进程中执行 Python 代码。

    Args:
        code: 要执行的 Python 代码字符串
        stdin: 写入子进程标准输入的内容
        timeout: 最大执行时间（秒）
        max_output_chars: stdout/stderr 截断长度

    Returns:
        ExecutionResult 包含 stdout, stderr, returncode, timed_out

    说明：

    - 本函数是通用执行入口，既可以给 stdin/stdout 程序使用，也可以给“代码+测试断言”
      这种模式使用。
    - 当前实现是轻量 subprocess 隔离，不是强安全沙箱；后续如果切 Docker / remote
      runner，接口最好保持不变。
    """
    # 创建临时 .py 文件，将代码字符串写入磁盘，以便用 subprocess 启动独立 python3 进程执行。
    # - mode="w": 文本写入模式
    # - suffix=".py": 文件后缀为 .py，使其可被 python3 直接执行
    # - delete=False: with 块退出时不自动删除文件，因为后续 subprocess.run 还需要读取该文件；
    #   文件会在 finally 块中通过 os.unlink(tmp_path) 手动清理
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp:
        tmp.write(code)          # 将待执行的代码写入临时文件
        tmp_path = tmp.name      # 保存临时文件的完整路径（如 /tmp/tmpXXXXXX.py）

    try:
        # subprocess.run() 是 Python 3.5+ 推荐的执行外部命令的高级接口。
        # 它会启动一个子进程，等待其结束，然后返回 CompletedProcess 对象。
        #
        # 常见用法对比：
        #   subprocess.run(...)          — 同步执行，等待子进程结束（本文件使用的方式）
        #   subprocess.Popen(...)        — 更底层，可异步交互（读写 stdin/stdout 流）
        #   subprocess.call(...)         — 旧接口，仅返回 returncode，已被 run() 取代
        #   subprocess.check_output(...) — 旧接口，返回 stdout，出错时抛异常
        #
        # 参数说明：
        #   ["python3", tmp_path]  — 要执行的命令，等价于 shell 中的 `python3 /tmp/tmpXXXXXX.py`
        #   capture_output=True    — 捕获 stdout 和 stderr（等价于 stdout=PIPE, stderr=PIPE）
        #   text=True              — 将 stdout/stderr 以 str 返回而非 bytes
        #   timeout=timeout        — 超时秒数，超时则抛出 subprocess.TimeoutExpired
        #   env={...}              — 子进程的环境变量，这里继承当前环境并额外设置
        #                            PYTHONDONTWRITEBYTECODE=1 来禁止生成 .pyc 缓存文件
        #
        # 返回值 proc (CompletedProcess) 的主要属性：
        #   proc.stdout      — 子进程标准输出（str）
        #   proc.stderr      — 子进程标准错误（str）
        #   proc.returncode  — 退出码，0 表示成功
        started_at = time.monotonic()
        proc = subprocess.run(
            [sys.executable, tmp_path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        runtime_seconds = time.monotonic() - started_at
        return ExecutionResult(
            stdout=proc.stdout[:max_output_chars],
            stderr=proc.stderr[:max_output_chars],
            returncode=proc.returncode,
            timed_out=False,
            runtime_seconds=runtime_seconds,
        )
    except subprocess.TimeoutExpired as e:
        # TimeoutExpired 可能携带部分 stdout/stderr。这里尽量保留它们，便于上层调试。
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return ExecutionResult(
            stdout=stdout[:max_output_chars],
            stderr=(stderr or f"Execution timed out after {timeout}s")[:max_output_chars],
            returncode=-1,
            timed_out=True,
            runtime_seconds=float(timeout),
        )
    except Exception as e:
        # 这里捕获的是“环境执行失败”，例如子进程拉起异常，而不是用户代码中的 RE。
        # 用户代码中的异常会体现在 returncode != 0 和 stderr 中。
        return ExecutionResult(
            stdout="",
            stderr=f"Execution error: {e}",
            returncode=-1,
            timed_out=False,
            runtime_seconds=0.0,
        )
    finally:
        # 从文件系统上删除这个临时文件
        os.unlink(tmp_path)


def execute_stdio(
    code: str,
    stdin: str,
    timeout: int | float = 5,
    max_output_chars: int = 4096,
) -> ExecutionResult:
    """执行完整 stdin/stdout Python 程序。

    这是 OJ-like v1 主链路真正使用的入口。judge 层会把 `OJTestCase.input`
    作为 `stdin` 传进来，再根据返回的 stdout/stderr/returncode 决定 verdict。
    """
    return execute_code(
        code=code,
        stdin=stdin,
        timeout=timeout,
        max_output_chars=max_output_chars,
    )
