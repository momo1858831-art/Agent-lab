from typing import Optional,Dict,Any,Literal,List
from datetime import datetime,timezone
from pydantic import BaseModel

# 限制消息角色的类型
MessageRole=Literal["user","assistant","system","tool"]

class Message(BaseModel):

    # 数据规则模板
    id:int
    content:Optional[str]=None
    role:MessageRole
    timestamp:datetime
    metadata:Optional[Dict[str,Any]]=None
    tool_calls:Optional[List[Dict[str,Any]]]=None # 工具调用信息 如调用工具的id,name,参数等 role=assistant的专属
    tool_call_id:Optional[str]=None # 工具id role=tool的专属

    def __init__(self,id:int,role:MessageRole,content:Optional[str]=None,**kwargs):
        tool_calls=kwargs.get("tool_calls")
        tool_call_id=kwargs.get("tool_call_id")
        if role=="tool" and tool_call_id is None:
            raise ValueError("tool消息必须包含tool_call_id字段")
        if tool_calls is not None and role!="assistant":
            raise ValueError("tool_calls只能出现在assistant消息中")
        if tool_call_id is not None and role!="tool":
            raise ValueError("tool_call_id只能出现在tool消息中")
        # LLM既不回答用户 也不调用工具
        if role=="assistant" and content is None and tool_calls is None:
            raise ValueError("无意义回答")
        if role in ("user","system","tool") and content is None:
            raise ValueError(f"消息{role}必须包含content字段")
        super().__init__(
            id=id,
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp",datetime.now(timezone.utc)),
            metadata=kwargs.get("metadata",{}),
            tool_calls=kwargs.get("tool_calls"),
            tool_call_id=kwargs.get("tool_call_id")
        )
    
    # 转换为字典格式
    def to_dict(self):
        message={
            "role":self.role,
            "content":self.content
        }
        if self.tool_calls is not None:
            message["tool_calls"]=self.tool_calls
        if self.tool_call_id is not None:
            message["tool_call_id"]=self.tool_call_id
        return message

    # 打印
    def __str__(self):
        return f"[{self.role}] {self.content}"

if __name__ == "__main__":
    message=Message(
        id=2,
        role="tool",
        content="搜索结果",
        tool_call_id="call_123"
    )
    print(message)
    print(message.to_dict())
