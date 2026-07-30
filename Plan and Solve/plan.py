import re
import ast
from LLM import LLM


# plan提示词模板
PLANNER_PROMPT_TEMPLATE="""
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表,其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

class Planner:

    def __init__(self,llm):
        self.llm=llm
    
    # 生成计划
    def plan(self,question:str):
        # 提示词
        prompt=PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
        print("正在生成计划")
        response_text=self.llm.think(messages=messages) or ""
        print(f"计划已生成:\n")
        # 解析LLM输出
        try:
            # 找出'''python与'''之间的内容
            match=re.search(
                # ?:固定搭配 使()的内容不占用group编号 但匹配结果仍包含此项
                r"```python\s*(.*?)```",
                response_text,
                re.DOTALL
            )
            plan_str=match.group(1) if match else ""
            # 将字符串转化为py列表
            # 寻找[作为列表开始寻找之处 每个""里的内容就是list的一项 引号之外的,表示分隔 以]结束
            plan=ast.literal_eval(plan_str)
            return plan if isinstance(plan,list) else []
        except Exception as e:
            print(f"解析时发生错误,错误原因为:{e}")
            return []
            
