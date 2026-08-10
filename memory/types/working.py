from datetime import datetime,timedelta
from typing import Dict,List,Any
from ..base import BaseMemory,MemoryConfig,MemoryItem
import heapq
import tiktoken

# 会话级短记忆
class WorkingMemory(BaseMemory):

    def __init__(self,config:MemoryConfig,storage_backend=None):
        super().__init__(config,storage_backend)
        self.encoding = tiktoken.get_encoding("o200k_base")
        # 工作记忆特定配置
        self.max_capacity=self.config.working_memory_capacity
        self.max_tokens=self.config.working_memory_tokens
        self.max_minutes=self.config.working_memory_ttl_minutes
        self.current_tokens=0
        self.session_start=datetime.now()
        self.memories:List[MemoryItem]=[]
        # 优先级队列管理记忆
        self.memory_heap=[] # 存储三元组( priority,memory_itrm.timesatmp,memory_item)

    # 计算token数
    def _count_tokens(self,content:str):
        return len(self.encoding.encode(content))

    # 更新时间衰减系数
    def _calculate_time_decay(self,timestamp:datetime):
        # 时间差 得到创建时长
        time_diff=datetime.now()-timestamp
        # 将时间差转化为小时
        hours_passed=time_diff.total_seconds()/3600
        # 时间越长 值越小 表示衰减的越多 
        decay_factor=self.config.decay_factor**(hours_passed/0.60206)
        # 最少保持0.1的权重
        return max(0.1,decay_factor)


    # 记忆优先级
    def _calculate_priority(self,memory:MemoryItem):
        priority=memory.importance
        # 更新时间衰减因子
        time_decay=self._calculate_time_decay(memory.timestamp)
        priority*=time_decay
        return priority

    # 过期清理
    def _expire_old_memories(self):
        # 按TTL清理过期记忆 并同步更新堆与token计数
        if not self.memories:
            return
        # 当前时间往前推self.max_minutes分钟
        cutoff_time=datetime.now()-timedelta(minutes=self.max_minutes)
        # 过滤保留的记忆
        kept:List[MemoryItem]=[]
        removed_token_sum=0
        for m in self.memories:
            if m.timestamp>=cutoff_time:
                kept.append(m)
            else:
                removed_token_sum+=self._count_tokens(m.content)
        # 更新历史记录与所用token数量
        self.memories=kept
        self.current_tokens-=removed_token_sum
        # 更新堆
        self.memory_heap=[]
        for m in self.memories:
            priority=self._calculate_priority(m)
            heapq.heappush(self.memory_heap,(priority,m.timestamp,m))

    # 从堆中删除
    def _remove_from_heap(self,memory_id:str):
        if not self.memory_heap:
            return
        new_heapq=[]
        for entry in self.memory_heap:
            memory=entry[2]
            if memory.id!=memory_id:
                new_heapq.append(entry)
        heapq.heapify(new_heapq)
        self.memory_heap=new_heapq

    # 删除指定id的记忆项
    def remove(self,memory_id:str):
        for i,memory in enumerate(self.memories):
            # 从列表删除
            if memory.id==memory_id:
                removed_memory=self.memories.pop(i)
                # 从堆删除
                self._remove_from_heap(memory_id)
                # 更新token数
                self.current_tokens-=self._count_tokens(removed_memory.content)
                return True
        return False

    # 删除最小优先级的记忆
    def _remove_lowest_priority_memory(self):
        if not self.memories:
            return
        lowest_priority=float('inf')
        lowest_memory=None
        for memory in self.memories:
            priority=self._calculate_priority(memory)
            if priority<lowest_priority:
                lowest_priority=priority
                lowest_memory=memory
        if lowest_memory:
            self.remove(lowest_memory.id)

    # 检查容量限制
    def _enforce_capacity_limits(self):
        # 最大容量
        while len(self.memories)>self.max_capacity:
            self._remove_lowest_priority_memory()
        # 最大token
        while self.current_tokens>self.max_tokens:
            self._remove_lowest_priority_memory()

    # 增添记忆项
    def add(self,memory_item:MemoryItem):
         # 过期清理
        self._expire_old_memories()
        # 判断能否加入
        importance=self._calculate_importance(memory_item.content,memory_item.importance)
        if importance<self.config.importance_threshold:
            raise ValueError("重要性得分小于阈值,无法加入历史记录")
        else:
            memory_item.importance=importance
        # 计算优先级
        priority=self._calculate_priority(memory_item)
        # 添加到最小堆中
        heapq.heappush(self.memory_heap,(priority,memory_item.timestamp,memory_item))
        self.memories.append(memory_item)
        # 更新tokens数
        self.current_tokens+=self._count_tokens(memory_item.content)
        # 检查容量、token数量限制
        self._enforce_capacity_limits()
        return memory_item.id

    # 更新
    def update(self,memory_id:str,content:str=None,importance:float=None,metadata:Dict[str,Any]=None):
        # 清理过时记忆
        self._expire_old_memories()
        for memory in self.memories:
            if memory_id==memory.id:
                old_tokens=self._count_tokens(memory.content)
                if importance is not None:
                    memory.importance=importance
                    self._remove_from_heap(memory_id)
                    priority=self._calculate_priority(memory)
                    heapq.heappush(self.memory_heap,(priority,memory.timestamp,memory))
                if content is not None:
                    memory.content=content
                    # 更新token数
                    new_tokens=self._count_tokens(content)
                    self.current_tokens=self.current_tokens-old_tokens+new_tokens
                    self._enforce_capacity_limits()
                if metadata is not None:
                    memory.metadata=metadata
                return True
        return False
        

            
