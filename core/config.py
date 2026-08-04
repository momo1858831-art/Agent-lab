import os
from typing import Optional,Dict,Any
from pydantic import BaseModel

class Config(BaseModel):

    # LLM配置
    model:str="gpt-5.6-luna"
    provider:str="openai"
    temperature:float=0
    max_tokens:int=10000

    # 系统配置
    debug:bool=False # 开发调试模数
    log_level:str="INFO" # 日志显示内容级别

    # 其它配置
    max_history_length:int=100

    # 类方法
    @classmethod
    def from_env(cls):
        # 从环境变量创建配置
        return cls(
            debug=os.getenv("debug","false").lower()=="true",
            log_level=os.getenv("log_level","INFO"),
            temperature=float(os.getenv("temperature",0)),
            max_tokens=int(os.getenv("max_tokens",10000))
        )

    # 转为字典
    def to_dict(self):
        return self.model_dump()