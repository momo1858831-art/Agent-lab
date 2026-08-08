from typing import Optional,Dict,Any,Literal,List
from datetime import datetime,timezone
from pydantic import BaseModel

# 限制消息角色的类型
MessageRole=Literal["user","assistant","system","tool"]

class Message(BaseModel):

    # 数据规则模板
    content:str
    role:MessageRole
    timestamp:datetime=None
    metadata:Optional[Dict[str,Any]]=None
    

    def __init__(self,role:MessageRole,content:str,**kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp",datetime.now(timezone.utc)),
            metadata=kwargs.get("metadata",{}),
        )
    
    # 转换为字典格式
    def to_dict(self):
        return {
            "role":self.role,
            "content":self.content
        }

    # 打印
    def __str__(self):
        return f"[{self.role}] {self.content}"


