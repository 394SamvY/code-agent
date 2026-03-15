from .state_manager import CodeEnvStateManager

__all__ = [
    "CodeEnvStateManager",
    "WriteCodeTool",
    "RunTestsTool",
    "SubmitTool",
]


def __getattr__(name):
    """延迟导入依赖 verl 的工具类，避免本地无 verl 时 import 失败。"""
    if name == "WriteCodeTool":
        from .write_code_tool import WriteCodeTool
        return WriteCodeTool
    if name == "RunTestsTool":
        from .run_tests_tool import RunTestsTool
        return RunTestsTool
    if name == "SubmitTool":
        from .submit_tool import SubmitTool
        return SubmitTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
