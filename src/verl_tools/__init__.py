__all__ = [
    "ExecuteCodeTool",
]


def __getattr__(name):
    """延迟导入依赖 verl 的工具类，避免本地无 verl 时 import 失败。"""
    if name == "ExecuteCodeTool":
        from .execute_code_tool import ExecuteCodeTool
        return ExecuteCodeTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
