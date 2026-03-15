"""
共享状态管理器
==============

三个 verl 工具 (write_code, run_tests, submit) 通过同一个 instance_id
共享同一个 CodeEnvironment 实例。
"""

from __future__ import annotations

import threading

from src.env.code_env import CodeEnvironment


class CodeEnvStateManager:
    """线程安全的环境状态管理器，按 instance_id 索引 CodeEnvironment 实例。"""

    _envs: dict[str, CodeEnvironment] = {}
    _lock = threading.Lock()

    @classmethod
    def create(
        cls,
        instance_id: str,
        test_list: list[str],
        entry_point: str = "",
        timeout: int = 5,
    ) -> None:
        with cls._lock:
            cls._envs[instance_id] = CodeEnvironment(
                problem_description="",
                test_list=test_list,
                entry_point=entry_point,
                timeout=timeout,
            )

    @classmethod
    def get(cls, instance_id: str) -> CodeEnvironment:
        return cls._envs[instance_id]

    @classmethod
    def exists(cls, instance_id: str) -> bool:
        return instance_id in cls._envs

    @classmethod
    def release(cls, instance_id: str) -> None:
        with cls._lock:
            cls._envs.pop(instance_id, None)
