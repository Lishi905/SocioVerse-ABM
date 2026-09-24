# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
简单示例: LLM集成Schelling模型

这个脚本展示了如何在Schelling分离模型中使用LLM。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from llm_integration import LLMDecisionMaker
from schelling_model import SchellingModel
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_rule_based():
    """
    示例1: 使用传统规则基础的Agent
    (不需要LLM，完全兼容)
    """
    print("\n" + "="*70)
    print("示例1: 规则基础Agent (传统方法)")
    print("="*70)
    
    # 创建模型 - 不使用LLM
    model = SchellingModel(
        height=30,
        width=30,
        density=0.8,
        minority_pc=0.5,
        homophily=5,
        use_llm=False  # 关闭LLM
    )
    
    print(f"✓ 创建了包含 {model.schedule.get_agent_count()} 个Agent的模型")
    print("✓ 所有Agent使用规则基础决策")
    
    # 运行仿真
    for step in range(20):
        model.step()
        if step % 5 == 0:
            print(f"  Step {step:2d}: Happy={model.happy:3d}, Converged={not model.running}")
        if not model.running:
            break
    
    print(f"✓ 仿真完成! 最终幸福度: {model.happy}/{model.schedule.get_agent_count()}")


def example_2_llm_based():
    """
    示例2: 使用LLM进行Agent决策
    (需要LLM服务可用)
    """
    print("\n" + "="*70)
    print("示例2: LLM-based Agent")
    print("="*70)
    
    try:
        # 初始化LLM决策制定者
        print("初始化LLM...")
        llm_maker = LLMDecisionMaker(
            provider_name="ollama",  # 使用本地Ollama (推荐)
            use_cache=True
        )
        print("✓ LLM初始化成功")
        
        # 创建模型 - 使用100% LLM
        model = SchellingModel(
            height=30,
            width=30,
            density=0.8,
            minority_pc=0.5,
            homophily=5,
            use_llm=True,                    # 启用LLM
            llm_decision_maker=llm_maker,
            llm_only_percentage=1.0          # 100% Agent使用LLM
        )
        
        print(f"✓ 创建了 {model.schedule.get_agent_count()} 个Agent")
        print(f"✓ LLM Agent数量: {model.llm_agents_count}")
        
        # 运行仿真
        for step in range(20):
            model.step()
            if step % 5 == 0:
                print(f"  Step {step:2d}: Happy={model.happy:3d}, LLM Calls cached")
            if not model.running:
                break
        
        print(f"✓ 仿真完成! 最终幸福度: {model.happy}/{model.schedule.get_agent_count()}")
        
    except Exception as e:
        print(f"✗ LLM初始化失败: {e}")
        print("提示: 请确保Ollama服务已启动或配置OpenAI API密钥")


def example_3_hybrid():
    """
    示例3: 混合Agent (50% LLM + 50% 规则基础)
    """
    print("\n" + "="*70)
    print("示例3: 混合Agent (50% LLM + 50% 规则)")
    print("="*70)
    
    try:
        # 初始化LLM
        llm_maker = LLMDecisionMaker(
            provider_name="ollama",
            use_cache=True
        )
        
        # 创建混合模型
        model = SchellingModel(
            height=30,
            width=30,
            density=0.8,
            minority_pc=0.5,
            homophily=5,
            use_llm=True,
            llm_decision_maker=llm_maker,
            llm_only_percentage=0.5  # 50% 使用LLM
        )
        
        print(f"✓ 创建了 {model.schedule.get_agent_count()} 个Agent")
        print(f"  - LLM Agent: {model.llm_agents_count} ({100*model.llm_agents_count/model.schedule.get_agent_count():.1f}%)")
        print(f"  - 规则Agent: {model.schedule.get_agent_count() - model.llm_agents_count} ({100*(model.schedule.get_agent_count() - model.llm_agents_count)/model.schedule.get_agent_count():.1f}%)")
        
        # 运行仿真
        for step in range(20):
            model.step()
            if step % 5 == 0:
                print(f"  Step {step:2d}: Happy={model.happy:3d}")
            if not model.running:
                break
        
        print(f"✓ 仿真完成!")
        
    except Exception as e:
        print(f"✗ 混合模型失败: {e}")


def example_4_fallback():
    """
    示例4: 优雅降级 - LLM失败时自动回退规则
    """
    print("\n" + "="*70)
    print("示例4: 优雅降级 (LLM失败 -> 自动回退规则)")
    print("="*70)
    
    try:
        # 尝试使用不存在的LLM服务
        print("尝试连接不存在的LLM服务...")
        llm_maker = LLMDecisionMaker(
            provider_name="ollama",
            use_cache=False
        )
        
        print("✗ 这个例子需要LLM失败才能演示降级")
        print("  在实际使用中，如果LLM失败，系统会自动回退到规则决策")
        
    except Exception as e:
        print(f"✓ 捕获到错误 (符合预期): {type(e).__name__}")
        print("✓ 在实际使用中会自动回退到规则决策")
        
        # 自动使用规则基础
        print("\n改用规则基础...")
        model = SchellingModel(
            height=30,
            width=30,
            density=0.8,
            minority_pc=0.5,
            homophily=5,
            use_llm=False
        )
        
        # 运行仿真
        for step in range(20):
            model.step()
            if step % 5 == 0:
                print(f"  Step {step:2d}: Happy={model.happy:3d} (使用规则决策)")
            if not model.running:
                break
        
        print(f"✓ 降级仿真完成!")


def example_5_custom_prompt():
    """
    示例5: 自定义提示词
    展示如何修改LLM的决策提示
    """
    print("\n" + "="*70)
    print("示例5: 自定义提示词")
    print("="*70)
    
    print("""
当前的LLM提示词包括:
- Agent的身份(类型0或1)
- 邻居情况(相同邻居数、总邻居数、相似性比例)
- 容忍度阈值
- 可用房间数

要自定义提示词:
1. 打开 llm_integration.py
2. 找到 _construct_prompt() 方法
3. 修改提示词模板

例如，可以添加:
- 社会压力因素
- 长期幸福度目标
- 随机风险容忍度
    """)


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█ Schelling分离模型 - LLM集成示例")
    print("█"*70)
    
    # 运行示例
    example_1_rule_based()      # 总是可以运行
    example_3_hybrid()          # 需要LLM
    example_5_custom_prompt()   # 信息示例
    
    print("\n" + "="*70)
    print("所有示例完成!")
    print("="*70)
    print("\n下一步:")
    print("1. 查看 QUICKSTART.md 进行快速开始")
    print("2. 查看 LLM_INTEGRATION_GUIDE.md 了解详细信息")
    print("3. 运行 python run_llm_simulation.py 进行完整实验")
    print()
