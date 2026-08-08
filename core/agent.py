from abc import ABC,abstractmethod
from typing import Optional,Any
from ..memory.message import Message
from .config import Config
from .LLM_extension import LLMExtension

# 抽象基类
class Agent(ABC):

    def __init__(self,name:str,llm:LLMExtension,system_prompt:Optional[str]=None,config:Optional[Config]=None):
        self.name=name
        self.llm=llm
        self.system_prompt=system_prompt
        self.config=config or Config()
        self.history:list[Message]=[]

    # 标记一个方法必须由子类实现
    @abstractmethod
    def run(self,input_text:str,**kwargs):
        # 运行Agent
        pass

    # 添加历史记录
    def add_message(self,message:Message):
        self.history.append(message)

    # 清除历史记录
    def clear_history(self):
        self.history.clear()

    # 获取历史记录
    def get_history(self):
        # 使用copy防止改变接受值对history本身产生影响
        return self.history.copy()

    def __str__(self):
        return f"Agent(name={self.name},provider={self.llm.provider})"