from abc import ABC,abstractmethod
from typing import Dict,Any,List


class Tool(ABC):

    def __init__(self,name:str,description:str):
        self.name=name
        self.description=description

    # 执行工具
    @abstractmethod
    def run(self,parameters:Dict[str,Any]):
        pass

    # 获取工具参数定义
    @abstractmethod
    def get_parameters(self):
        pass

    # 转换为openai function calling schema格式
    def to_open_schema(self):
        parameters=self.get_parameters()
        properties={}
        required=[]
        for param in parameters:
            prop={
                "type":param.type,
                "description":param.description
            }
            if param.default is not None:
                prop["description"]=f"{param.description}(默认{param.default})"
            if param.type=="array":
                prop["items"]={"type":"string"}
            properties[param.name]=prop
            if param.required:
                required.append(param.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        