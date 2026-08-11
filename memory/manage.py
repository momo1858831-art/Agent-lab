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

    # 删除记忆
    def remove_memory(self,memory_id:str):
        # 因为使用的是uuid 所以此id只会出现在一种记忆中 可直接return
        for memory_type,memory_instance in self.memory_types.items():
            if memory_instance.has_memory(memory_id):
                return memory_instance.remove(memory_id)
        logger.warning(f"未找到记忆:{memory_id}")
        return False

    # 更新记忆
    def update_memory(self,memory_id:str,content:str=None,importance:float=None,metadata:Dict[str,Any]=None):
        for memory_type,memory_instance in self.memory_types.items():
            if memory_instance.has_memory(memory_id):
                return memory_instance.update(memory_id,content,importance,metadata)
        logger.warning(f"未找到记忆:{memory_id}")
        return False

    # 整合记忆
    def consolidate_memories(self,from_type:str="working",to_type:str="episodic",importance_threshold:float=0.7):
        """
            from_type 源记忆类型
            to_type:目标记忆类型
            importance_threshold:重要性阈值
        """
        if not 0<=importance_threshold<=1:
            raise ValueError("importance_threshold应在0到1之间")
        if from_type==to_type:
            logger.warning("记忆类型相同,无需整合")
            return 0
        if from_type not in self.memory_types or to_type not in self.memory_types:
            logger.warning(f"记忆类型不存在:{from_type}->{to_type}")
            return 0
        source_memory=self.memory_types[from_type]
        target_memory=self.memory_types[to_type]
        # 获取需要整合的记忆
        all_memories=source_memory.get_all()
        # 重要性阈值过滤
        candidates=[
            m for m in all_memories if m.importance>=importance_threshold
        ]
        # 移动记忆数量
        consolidated_count=0
        for memory in candidates:
            # 移动到目标类型记忆
            if source_memory.remove(memory.id):
                memory.memory_type=to_type
                memory.importance=min(1.0,1.1*memory.importance) # 提高重要性
                target_memory.add(memory)
                consolidated_count+=1
        logger.info(f"记忆整合完成:已将{consolidated_count}条记忆从{from_type}移动到{to_type}")
        return consolidated_count

    # 获取记忆统计信息
    def get_memory_stats(self):
        stats={
            "user_id":self.user_id,
            "enabled_types":list(self.memory_types.keys()),
            "total_memories":0,
            "memories_by_type":{},
            "config":{
                "max_capacity":self.config.max_capacity,
                "importance_threshold":self.config.importance_threshold,
                "decay_factor":self.config.decay_factor
            }
        }
        for memory_type,memory_instance in self.memory_types.items():
            type_stats=memory_instance.get_stats()
            stats["memories_by_type"][memory_type]=type_stats
            stats["total_memories"]+=type_stats.get("count",0)
        return stats

    # 清空所有记忆
    def clear_all_memories(self):
        for memory_type,memory_instance in self.memory_types.items():
            memory_instance.clear()
        logger.info("所有记忆已清空")

    # 判断是否为情景记忆内容
    def _is_episodic_content(self,content:str):
        episodic_keywords=["昨天","今天","明天","上次","记得","发生","经历"]
        return any(keyword in content for keyword in episodic_keywords)

    # 判断是否为语义记忆内容
    def _is_semantic_content(self,content:str):
        semantic_keywords=["定义","概念","规则","知识","原理","方法"]
        return any(keyword in content for keyword in semantic_keywords)

    def __str__(self):
        stats=self.get_memory_stats()
        return f"MemoryManager(user={self.user_id},total={stats['total_memories']})"

    