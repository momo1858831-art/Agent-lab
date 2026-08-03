from LLM import LLM

# solve提示词模板
EXECUTOR_PROMPT_TEMPLATE="""
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划,一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

class Exceutor:

    def __init__(self,llm):
        self.llm=llm

    # 根据相关信息处理每个步骤
    def exceute(self,question:str,plan:list[str]):
        history=[]
        print("---正在执行计划---\n")
        for i,step in enumerate(plan):
            print(f"正在执行步骤{i+1}/{len(plan)}:{step}")
            prompt=EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ]
            print(f"步骤{i+1}已完成,结果为:")
            response_text=self.llm.think(messages=messages) or ""
            print()
            history+=f"步骤{i+1}:{step}\n结果:{response_text}\n\n"
            # print(f"步骤{i+1}已完成,结果为:")
        final_answer=response_text
        return final_answer