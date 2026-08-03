from .plan import Planner
from .solve import Exceutor
from LLM import LLM

class PlanAndSolveAgent:

    def __init__(self,llm):
        self.llm=llm
        self.planner=Planner(self.llm)
        self.exceutor=Exceutor(self.llm)

    def run(self,question:str):
        print(f"\n---正在处理问题:\n{question}---\n")
        plan=self.planner.plan(question)
        if not plan:
            print("无法生成相应计划,任务终止")
            return
        final_answer=self.exceutor.exceute(question,plan)
        print(f"\n任务完成,最终答案为{final_answer}")

# python -m "Plan and Solve.plan_and_solve" base文件夹下
if __name__=='__main__':
    llm=LLM()
    agent=PlanAndSolveAgent(llm)
    question="""
        一家咖啡店预计周六售出 120 杯咖啡，周日售出 150 杯。
        每杯需要 18 克咖啡豆和 200 毫升牛奶。为应对需求波动，咖啡豆和牛奶均需额外准备 10%。
        咖啡豆按每袋 1 千克购买，每袋 80 元；牛奶按每盒 1 升购买，每盒 12 元，且均只能购买整数袋/盒。
        预算为 1,200 元。需要购买多少袋咖啡豆、多少盒牛奶？总成本是多少？是否超出预算？
    """
    agent.run(question)