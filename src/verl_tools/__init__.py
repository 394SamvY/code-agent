__all__ = [
    "RunPublicTestsTool",
    "SubmitSolutionTool",
]


def __getattr__(name):
    """延迟导入依赖 verl 的工具类，避免本地无 verl 时 import 失败。"""
    if name == "RunPublicTestsTool":
        from .oj_tools import RunPublicTestsTool
        return RunPublicTestsTool
    if name == "SubmitSolutionTool":
        from .oj_tools import SubmitSolutionTool
        return SubmitSolutionTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
