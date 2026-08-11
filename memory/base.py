from pydantic import BaseModel
from datetime import datetime
from typing import Dict,Any,List
from abc import ABC,abstractmethod
import uuid

# 记忆项
class MemoryItem(BaseModel):
    id:str
    content:str
    memory_type:str
    user_id:str
    timestamp:datetime
    importance:float=0.5
    metadata:Dict[str,Any]
    
    class Config:
        arbitrary_types_allowed=True

# 记忆系统配置
class MemoryConfig(BaseModel):
    storage_path:str="./memory_data" # 存储路径
    max_capacity:int=100
    importance_threshold:float=0.1 # 重要性阈值
    decay_factor:float=0.5 # 记忆衰减系数
    # 工作记忆特定配置
    working_memory_capacity:int=10
    working_memory_tokens:int=2000
    working_memory_ttl_minutes:int=120 # 工作记忆存活时长
    # 感知记忆特定配置 支持的模态,如文本、图片、音频、视频
    perceptual_memory_modalities:List[str]=["text","image","audio","video"]

# 记忆基类
class BaseMemory(ABC):

    def __init__(self,config:MemoryConfig,storage_backend=None):
        self.config=config
        self.storage=storage_backend # 记忆存储到哪里
        # 取当前类名 转化为小写 将所有memory替换为""
        self.memory_type=self.__class__.__name__.lower().replace("memory","")

    @abstractmethod
    def add(self,memory_item:MemoryItem):
        pass

    @abstractmethod
    def retrieve(self,query:str,limit:int=5,**kwargs):
        """
            query:查询内容
            limit:返回数量限制
            **kwargs:其它检索参数
        """
        pass

    @abstractmethod
    def update(self,memory_id:str,content:str=None,importance:float=None,metadata:Dict[str,Any]=None):
        pass

    @abstractmethod
    def remove(self,memory_id:str):
        pass

    @abstractmethod
    def has_memory(self,memory_id:str):
        # 检查记忆是否存在
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def get_stats(self):
        # 获取记忆统计信息
        pass

    def _generate_id(self):
        return str(uuid.uuid4())

    # 计算记忆重要性
    def _calculate_importance(self,content:str,base_importance:float=0.5):
        importance=base_importance
        # 基于内容长度
        if len(content)>100:
            importance+=0.1
        # 基于关键词
        importance_keywords=["重要","关键","必须","注意","警告","错误"]
        if any(keyword in content for keyword in importance_keywords):
            importance+=0.2
        return max(0.0,min(1.0,importance))

    # 打印各种记忆系统的记忆数
    def __str__(self):
        stats=self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count',0)})"

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def get_all(self):
        pass

