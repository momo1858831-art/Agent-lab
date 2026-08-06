from pydantic import BaseModel
from typing import Any

class ToolParameter(BaseModel):
    """工具需要的参数定义"""
    name:str # 参数名
    type:str # 参数类型
    description:str # 参数功能描述
    required:bool=True # 是否必须传
    default:Any=None # 未传入时的默认值