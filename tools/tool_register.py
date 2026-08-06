from .tool import Tool
from typing import Any,Callable

class Toolregistr():

    def __init__(self):
        self.tools:dict[str,Tool]={}
        self.functions:dict[str,dict[str,Any]]={}

    def register_tool(self,tool:Tool):
        if tool.name in self.tools:
            print(f"工具{tool.name}功能已被覆盖")
        self.tools[tool.name]=tool
        print(f"工具{tool.name}已注册")

    def register_function(self,name:str,description:str,func:Callable[[str],str]):
        """
        name:工具名称
        description:工具描述
        func:工具函数,这个函数接受字符串类型参数,返回字符串结果
        """
        if name in self.functions:
            print(f"工具{name}功能已被覆盖")
        self.functions[name]={
            "description":description,
            "func":func
        }
        print(f"工具{name}已注册")

    def get_tools_description(self):
        descriptions=[]
        # 取字典中的value值
        for tool in self.tools.values():
            descriptions.append(f"{tool.name}:{tool.description}")
        # 同时取key和value
        for name,info in self.functions.items():
            descriptions.append(f"{name}:{info['description']}")
        return "\n".join(descriptions) if descriptions else "暂无可调用工具"
