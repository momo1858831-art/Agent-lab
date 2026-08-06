from pydantic import BaseModel
from typing import Any

class ToolParameter(BaseModel):
    """工具参数定义"""
    name:str
    type:str
    description:str
    required:bool=True
    default:Any=None