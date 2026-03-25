"""
verl 工具逻辑本地验证
=====================

不需要 GPU，不需要 verl 安装，纯逻辑测试：
- 通过 CodeEnvironment.execute_tool() 执行单工具
- Reward 计算逻辑
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.code_env import CodeEnvironment


def test_single_tool_pass():
    """测试单工具：代码正确，一次通过。"""
    env = CodeEnvironment(
        problem_description="Write a function add(a, b)",
        test_list=["assert add(1,2)==3", "assert add(0,0)==0"],
    )

    result = env.execute_tool("execute_code", code="def add(a, b):\n    return a + b")
    assert "2/2 tests passed" in result
    assert "All tests passed" in result
    assert env.current_code == "def add(a, b):\n    return a + b"
    assert env.tool_history == ["execute_code"]
    assert env.test_results_history == [{"passed": 2, "total": 2}]
    assert env.is_all_passed is True

    print("[PASS] test_single_tool_pass")


def test_single_tool_syntax_error():
    """测试单工具：语法错误。"""
    env = CodeEnvironment(
        problem_description="Write a function",
        test_list=["assert f(1)==1"],
    )

    result = env.execute_tool("execute_code", code="def f(x)\n    return x")
    assert "Syntax error" in result
    # 语法错误不记录到 test_results_history（未执行测试）
    assert env.test_results_history == []

    print("[PASS] test_single_tool_syntax_error")


def test_single_tool_bug_fix():
    """测试单工具：先失败 → 修复 → 通过。"""
    env = CodeEnvironment(
        problem_description="Write a function multiply(a, b)",
        test_list=["assert multiply(3,4)==12"],
    )

    # 第一次：错误代码
    result = env.execute_tool("execute_code", code="def multiply(a, b):\n    return a + b")
    assert "0/1 tests passed" in result
    assert "Failure details" in result
    assert env.test_results_history == [{"passed": 0, "total": 1}]

    # 第二次：修复后
    result = env.execute_tool("execute_code", code="def multiply(a, b):\n    return a * b")
    assert "1/1 tests passed" in result
    assert "All tests passed" in result
    assert env.test_results_history == [{"passed": 0, "total": 1}, {"passed": 1, "total": 1}]
    assert env.is_all_passed is True

    print("[PASS] test_single_tool_bug_fix")


def test_single_tool_partial_pass():
    """测试单工具：部分通过。"""
    env = CodeEnvironment(
        problem_description="Write f(x)",
        test_list=["assert f(1)==1", "assert f(2)==4", "assert f(3)==8"],
    )

    result = env.execute_tool("execute_code", code="def f(x):\n    return x * x")
    # f(1)=1 pass, f(2)=4 pass, f(3)=9!=8 fail → 2/3
    assert "2/3 tests passed" in result
    assert env.last_pass_ratio == 2 / 3

    print("[PASS] test_single_tool_partial_pass")


def test_reward_calculation():
    """测试 reward 计算逻辑（模拟 ExecuteCodeTool.calc_reward）。"""

    def calc_reward(attempts: list[tuple[int, int]]) -> float:
        """从 ExecuteCodeTool.calc_reward 中提取的纯逻辑版本。"""
        if not attempts:
            return 0.0

        last_passed, last_total = attempts[-1]
        exec_reward = last_passed / last_total if last_total > 0 else 0.0

        fix_reward = 0.0
        if len(attempts) >= 2:
            first_passed, first_total = attempts[0]
            if first_passed < first_total and last_passed == last_total:
                fix_reward = 0.2

        return exec_reward + fix_reward

    # Case 1: 一次通过
    r1 = calc_reward([(3, 3)])
    assert r1 == 1.0, f"Expected 1.0, got {r1}"

    # Case 2: 修复后通过
    r2 = calc_reward([(0, 1), (1, 1)])
    assert r2 == 1.0 + 0.2, f"Expected 1.2, got {r2}"

    # Case 3: 多次修复后通过
    r3 = calc_reward([(0, 3), (1, 3), (3, 3)])
    assert r3 == 1.0 + 0.2, f"Expected 1.2, got {r3}"

    # Case 4: 始终未通过
    r4 = calc_reward([(0, 3), (1, 3)])
    expected4 = 1 / 3
    assert abs(r4 - expected4) < 1e-9, f"Expected {expected4}, got {r4}"

    # Case 5: 空轨迹
    r5 = calc_reward([])
    assert r5 == 0.0, f"Expected 0.0, got {r5}"

    print("[PASS] test_reward_calculation")


def test_unknown_tool():
    """测试调用不存在的工具。"""
    env = CodeEnvironment(
        problem_description="test",
        test_list=["assert True"],
    )

    result = env.execute_tool("write_code", code="pass")
    assert "Error: Unknown tool" in result

    print("[PASS] test_unknown_tool")


if __name__ == "__main__":
    test_single_tool_pass()
    test_single_tool_syntax_error()
    test_single_tool_bug_fix()
    test_single_tool_partial_pass()
    test_reward_calculation()
    test_unknown_tool()
    print("\nAll tests passed!")
