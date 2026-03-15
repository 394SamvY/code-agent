"""
verl 工具逻辑本地验证
=====================

不需要 GPU，不需要 verl 安装，纯逻辑测试：
- StateManager CRUD（存 CodeEnvironment 实例）
- 通过 CodeEnvironment.execute_tool() 执行工具
- Reward 计算逻辑
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.code_env import CodeEnvironment
from src.verl_tools.state_manager import CodeEnvStateManager


def test_state_manager():
    """测试 StateManager 的 CRUD 操作（存 CodeEnvironment 实例）。"""
    instance_id = "test-instance-001"

    CodeEnvStateManager.create(
        instance_id,
        test_list=["assert add(1,2)==3"],
        entry_point="add",
        timeout=5,
    )
    assert CodeEnvStateManager.exists(instance_id)

    env = CodeEnvStateManager.get(instance_id)
    assert isinstance(env, CodeEnvironment)
    assert env.current_code == ""
    assert env.tool_history == []
    assert env.test_results_history == []
    assert env.is_done is False

    CodeEnvStateManager.release(instance_id)
    assert not CodeEnvStateManager.exists(instance_id)
    print("[PASS] test_state_manager")


def test_tool_execution_flow():
    """测试完整的 write_code -> run_tests -> submit 流程。"""
    env = CodeEnvironment(
        problem_description="Write a function add(a, b)",
        test_list=["assert add(1,2)==3", "assert add(0,0)==0"],
    )

    result = env.execute_tool("write_code", code="def add(a, b):\n    return a + b")
    assert "Syntax OK" in result
    assert env.current_code == "def add(a, b):\n    return a + b"
    assert env.tool_history == ["write_code"]

    result = env.execute_tool("run_tests")
    assert "2 passed" in result
    assert "0 failed" in result
    assert env.tool_history == ["write_code", "run_tests"]
    assert env.test_results_history == [{"passed": 2, "total": 2}]

    result = env.execute_tool("submit")
    assert "Accepted" in result
    assert env.submit_passed is True
    assert env.submit_pass_ratio == 1.0
    assert env.tool_history == ["write_code", "run_tests", "submit"]

    print("[PASS] test_tool_execution_flow")


def test_tool_execution_with_bug_fix():
    """测试 先失败 -> 修复 -> 通过 的流程。"""
    env = CodeEnvironment(
        problem_description="Write a function multiply(a, b)",
        test_list=["assert multiply(3,4)==12"],
    )

    env.execute_tool("write_code", code="def multiply(a, b):\n    return a + b")
    assert env.tool_history == ["write_code"]

    result = env.execute_tool("run_tests")
    assert "0 passed" in result or "failed" in result
    assert env.test_results_history == [{"passed": 0, "total": 1}]

    env.execute_tool("write_code", code="def multiply(a, b):\n    return a * b")

    result = env.execute_tool("run_tests")
    assert "1 passed" in result
    assert env.test_results_history[-1] == {"passed": 1, "total": 1}

    result = env.execute_tool("submit")
    assert "Accepted" in result
    assert env.submit_passed is True
    assert env.submit_pass_ratio == 1.0

    print("[PASS] test_tool_execution_with_bug_fix")


def test_reward_calculation():
    """测试 reward 计算逻辑（不依赖 verl 导入）。"""

    def calc_reward(env: CodeEnvironment) -> float:
        """从 SubmitTool.calc_reward 中提取的纯逻辑版本。"""
        history = env.tool_history
        test_history = env.test_results_history

        exec_reward = env.submit_pass_ratio

        order_reward = 0.0
        wc_indices = [i for i, t in enumerate(history) if t == "write_code"]
        rt_indices = [i for i, t in enumerate(history) if t == "run_tests"]
        if wc_indices and rt_indices and wc_indices[0] < rt_indices[0]:
            order_reward = 0.1

        fix_reward = 0.0
        if len(test_history) >= 2:
            first = test_history[0]
            last = test_history[-1]
            if first["passed"] < first["total"] and last["passed"] == last["total"]:
                fix_reward = 0.2

        submit_reward = 0.0
        if env.submit_passed and "submit" in history:
            submit_reward = 0.1

        return exec_reward + order_reward + fix_reward + submit_reward

    # Case 1: 完美轨迹 (write -> test pass -> submit)
    env1 = CodeEnvironment("", ["assert add(1,2)==3", "assert add(0,0)==0", "assert add(-1,1)==0"])
    env1.execute_tool("write_code", code="def add(a, b):\n    return a + b")
    env1.execute_tool("run_tests")
    env1.execute_tool("submit")
    r1 = calc_reward(env1)
    assert r1 == 1.0 + 0.1 + 0.0 + 0.1, f"Expected 1.2, got {r1}"

    # Case 2: 修复轨迹 (write -> test fail -> write -> test pass -> submit)
    env2 = CodeEnvironment("", ["assert multiply(3,4)==12"])
    env2.execute_tool("write_code", code="def multiply(a, b):\n    return a + b")
    env2.execute_tool("run_tests")
    env2.execute_tool("write_code", code="def multiply(a, b):\n    return a * b")
    env2.execute_tool("run_tests")
    env2.execute_tool("submit")
    r2 = calc_reward(env2)
    assert r2 == 1.0 + 0.1 + 0.2 + 0.1, f"Expected 1.4, got {r2}"

    # Case 3: 空环境
    env3 = CodeEnvironment("", [])
    r3 = calc_reward(env3)
    assert r3 == 0.0, f"Expected 0.0, got {r3}"

    # Case 4: 部分通过 (2/3 tests pass)
    env4 = CodeEnvironment("", ["assert f(1)==1", "assert f(2)==4", "assert f(3)==8"])
    env4.execute_tool("write_code", code="def f(x):\n    return x * x")
    env4.execute_tool("run_tests")
    env4.execute_tool("submit")
    r4 = calc_reward(env4)
    expected4 = 2 / 3 + 0.1 + 0.0 + 0.0
    assert abs(r4 - expected4) < 1e-9, f"Expected {expected4}, got {r4}"

    # Case 5: 错误顺序 (run_tests 先于 write_code -> no order reward)
    env5 = CodeEnvironment("", ["assert add(1,2)==3"])
    env5.execute_tool("run_tests")  # 先调 run_tests（会报错，无代码）
    env5.execute_tool("write_code", code="def add(a, b):\n    return a + b")
    env5.execute_tool("run_tests")
    env5.execute_tool("submit")
    r5 = calc_reward(env5)
    # run_tests 先于 write_code，无 order_reward
    # test_results_history: [{"passed": 1, "total": 1}]（只有第二次成功的 run_tests 有记录，第一次因无代码提前返回）
    # 无 fix_reward（只有一次成功的测试记录）
    assert r5 == 1.0 + 0.0 + 0.0 + 0.1, f"Expected 1.1, got {r5}"

    print("[PASS] test_reward_calculation")


if __name__ == "__main__":
    test_state_manager()
    test_tool_execution_flow()
    test_tool_execution_with_bug_fix()
    test_reward_calculation()
    print("\nAll tests passed!")
