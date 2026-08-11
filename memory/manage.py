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
        # 默认重要性 与MemoryItem默认值保持一致
        if importance is None:
            importance=0.5
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


    # 检索记忆
    def retrieve_memories(self,query:str,memory_types:Optional[List[str]]=None,limit:int=10,min_importance:float=0.0,time_range:Optional[tuple]=None):
        """
            query:查询内容
            momories_types:要检索的记忆类型
            limit:返回数量限制
            min_importance:最小重要性阈值
            time_range:时间范围
        """
        # 若未指定则默认寻找所有记忆类型
        if memory_types is None:
            memory_types=list(self.memory_types.keys())
        if not memory_types:
            return []
        # 从各个记忆类型中探索
        all_results=[]
        # 每个类型的记忆最多返回的数量
        per_type_limit=max(1,limit//len(memory_types))
        for memory_type in memory_types:
            if memory_type in self.memory_types:
                memory_instance=self.memory_types[memory_type]
                try:
                    # 每个记忆类型使用自己的检索方法
                    types_results=memory_instance.retrieve(
                        query=query,
                        limit=per_type_limit,
                        user_id=self.user_id,
                        min_importance=min_importance,
                        time_range=time_range
                    )
                    all_results.extend(types_results)
                except ValueError:
                    raise
                except Exception as e:
                    logger.warning(f"检索{memory_type}记忆时出错{e}")
        return all_results[:limit]
        
    