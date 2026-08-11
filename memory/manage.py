from typing import Optional,Dict,List,Any
from .base import MemoryConfig,MemoryItem
from .types.working import WorkingMemory
import logging
import uuid
from datetime import datetime

logger=logging.getLogger(__name__)

# 记忆管理器
class MemoryManager:

    def __init__(
            self,
            config:Optional[MemoryConfig]=None,
            user_id:str="default_user",
            enable_working:bool=True,
    ):
        self.config=config or MemoryConfig()
        self.user_id=user_id
        # 初始化各记忆类型
        self.memory_types={}
        if enable_working:
            self.memory_types['working']=WorkingMemory(self.config)
        logger.info(f"MemoryManager初始化完成,可用记忆类型为:{list(self.memory_types.keys())}")

    def add_memory(self,content:str,memory_type:str="working",importance:Optional[float]=None,metadata:Optional[Dict[str,Any]]=None,auto_classify:bool=False):
        """
            content:记忆内容
            memory_type:记忆类型
            importance:重要性分数
            metadata:元数据
            auto_classify:是否自动分类到合适的记忆类型
        """
        # 自动分类记忆类型
        if auto_classify:
            memory_type=self._classify_memory_type(content,metadata)
        # 计算重要性
        if importance is None:
            importance=self._calculate_importance(content,metadata)
        # 创建记忆项
        memory_item=MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            user_id=self.user_id,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )
        # 添加到对应的记忆类型
        if memory_type in self.memory_types:
            memory_id=self.memory_types[memory_type].add(memory_item)
            logger.debug(f"已将记忆{memory_id}添加到{memory_type}")
            return memory_id
        else:
            raise ValueError(f"不支持的记忆类型")

    # 自动分类记忆类型
    def _classify_memory_type(self,content:str,metadata:Optional[Dict[str,Any]]):
        pass

    def _calculate_importance(self,content:str,metadata:Optional[Dict[str,Any]]=None):
        importance=0.5
        # 基于内容长度
        if len(content)>100:
            importance+=0.1
        # 基于关键词
        importance_keywords=["重要","关键","必须","注意","警告","错误"]
        if any(keyword in content for keyword in importance_keywords):
            importance+=0.2
        # 基于元数据
        if metadata:
            if metadata.get("priority")=="high":
                importance+=0.3
            elif metadata.get("priority")=="low":
                importance-=0.3
        return max(0.0,min(1.0,importance))
        
    