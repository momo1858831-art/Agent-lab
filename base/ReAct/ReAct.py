from .tool import ToolExecutor,search
from LLM import LLM
import re

# ReAct提示词模板 含占位符
REACT_PROMPT_TEMPLATE="""
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息,能够回答用户的最终问题时,必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


class ReActAgent:

    def __init__(self,llm:LLM,tool_exceutor:ToolExecutor,max_steps:int=5):
        self.llm=llm
        self.tool_exceutor=tool_exceutor
        self.max_steps=max_steps
        self.history=[]

    def _parse_output(self,text:str):
        # 从text提取Thought:后面的思考内容
        thought_match=re.search(
            # Thought: 固定匹配
            # \s* 任意长度的空白符(回车、空格、制表符等)
            # (.*?) 任意长度的任意字符 非贪婪
            # (?=\nAction:|$) 检查后面的内容是否为\nAction:(只检查不包含)或者已经到了文本末尾 ?=为特定搭配
            r"Thought:\s*(.*?)(?=\nAction:|$)",
            text,
            re.DOTALL # 让.可以匹配\n
        )
        # 从text提取Action:后面的内容
        action_match=re.search(
            r"Action:\s*(.*?)$",
            text,
            re.DOTALL
        )
        # 取匹配结果中第一个()匹配的内容
        thought=thought_match.group(1).strip() if thought_match else None
        action=action_match.group(1).strip() if action_match else None
        return thought,action

    # 提取工具名和工具输入
    def _parse_action(self,action_text:str):
        match=re.match(
            # (\w+) 匹配1个或多个单词字符(英文字母 数字 下划线)
            # \[ 匹配[ 因为[在正则表达式有特殊用处，因此需要转义符\ 同理]
            r"(\w+)\[(.*)\]",
            action_text,
            re.DOTALL
        )
        if match:
            return match.group(1),match.group(2)
        return None,None
        
    # 运行agent
    def run(self,question:str):
        self.history=[]
        current_step=0
        while current_step<self.max_steps:
            current_step+=1
            print(f"---当前为第{current_step}轮---")
            # 格式化提示词
            tools_desc=self.tool_exceutor.getAvailableTools()
            history_str="\n".join(self.history)
            # 替换模板中的占位符
            prompt=REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )
            # 调用LLM进行思考
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ]
            response_text=self.llm.think(messages=messages)
            if not response_text:
                print("LLM调用失败")
                break
            # 解析LLM输出
            thought,action=self._parse_output(response_text)
            if thought:
                print(f"\n思考:{thought}")
            if not action:
                print("\n警告:未能解析出有效的Action,流程终止")
                break
            # Action为Finish
            if action.startswith("Finish"):
                final_answer=re.match(r"Finish\[(.*)\]",action,re.DOTALL).group(1)
                print(f"最终答案为:{final_answer}")
                return final_answer
            # Action没结束继续执行
            tool_name,tool_input=self._parse_action(action)
            if not tool_name or not tool_input:
                continue;
            print(f"行动:{tool_name}:[{tool_input}]")
            tool_function=self.tool_exceutor.getTool(tool_name)
            if not tool_function:
                observation=f"错误,未找到名为'{tool_name}'的工具"
            else:
                observation=tool_function(tool_input)
            print(f"观察:{observation}")
            self.history.append(f"Action:{action}")
            self.history.append(f"observation:{observation}")
        print("已到达最终步数,流程终止")
        return None

if __name__=='__main__':
    llm=LLM()
    tool_exceutor=ToolExecutor()
    search_description="一个网页搜索引擎,当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具"
    tool_exceutor.registerTool("Search",search_description,search)
    agent=ReActAgent(llm,tool_exceutor)
    ans=agent.run("配置为13英寸、内存为1TB、运行内存为24GB的银色MacBook air怎么样")
    print(ans)

# python -m ReAct.ReAct base文件夹下