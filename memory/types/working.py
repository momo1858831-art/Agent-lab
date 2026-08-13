from datetime import datetime,timedelta
from typing import Dict,List,Any
from ..base import BaseMemory,MemoryConfig,MemoryItem
import heapq
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import os
import hashlib
import time
import uuid
import requests

load_dotenv()

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

    # 截断翻译文本
    def _truncate_translation_text(self,text:str):
        if len(text)<=20:
            return text
        return f"{text[:10]}{len(text)}{text[-10:]}"

    # 批量翻译英文
    def _translate_batch_to_english(self,texts:List[str],app_key:str,app_secret:str):
        # 生成签名参数
        salt=str(uuid.uuid4())
        current_time=str(int(time.time()))
        combined_text="".join(texts)
        # 待签名字符串
        sign_text=app_key+self._truncate_translation_text(combined_text)+salt+current_time+app_secret
        sign=hashlib.sha256(sign_text.encode("utf-8")).hexdigest()
        # 请求有道翻译
        response=requests.post(
            "https://openapi.youdao.com/v2/api",
            data={
                "q":texts,
                "from":"auto",
                "to":"en",
                "appKey":app_key,
                "salt":salt,
                "sign":sign,
                "signType":"v3",
                "curtime":current_time
            },
            timeout=15
        )
        response.raise_for_status()
        result=response.json()
        # 检查翻译结果
        if str(result.get("errorCode"))!="0":
            raise RuntimeError(f"有道翻译失败，错误码：{result.get('errorCode')}")
        translated_items=result.get("translateResults",[])
        if len(translated_items)!=len(texts):
            raise RuntimeError("翻译结果数量与输入数量不一致")
        return [item["translation"] for item in translated_items]

    # 将文本列表翻译为英文
    def _translate_to_english(self,documents:List[str]):
        app_key=os.getenv("translate_id")
        app_secret=os.getenv("translate_key")
        if not app_key or not app_secret:
            raise RuntimeError("缺少有道翻译 APP_KEY 或 APP_SECRET")
        if not documents:
            return []
        # 过滤空文本并保留原来的列表位置
        translated_documents=list(documents) # 浅拷贝 防止修改documents
        pending=[(index,text) for index,text in enumerate(documents) if text.strip()]
        # 有道单次最多5000字符 超过时自动分批
        batches=[]
        current_batch=[]
        current_length=0
        for index,text in pending:
            if len(text)>5000:
                raise ValueError("单条文本超过有道翻译的5000字符限制")
            # 假如当前批次+新文本长度大于5000
            if current_length+len(text)>5000:
                # 保存当前批次
                batches.append(current_batch)
                # 开始下一批次
                current_batch=[]
                current_length=0
            # 将文本添加到当前批次
            current_batch.append((index,text))
            current_length+=len(text)
        # 如果当前批次有内容(最后一批)
        if current_batch:
            batches.append(current_batch)
        # 分批翻译并恢复原来的列表顺序
        for batch in batches:
            indexes=[index for index,_ in batch]
            texts=[text for _,text in batch]
            # 批量翻译
            translations=self._translate_batch_to_english(texts,app_key,app_secret)
            # 配对
            for index,translation in zip(indexes,translations):
                translated_documents[index]=translation
        return translated_documents

    # 检索记忆
    def retrieve(self,query:str,limit:int=5,**kwargs):
        # 清理过时记忆
        self._expire_old_memories()
        if not self.memories:
            return []
        user_id=kwargs.get("user_id")
        # 按用户id过滤
        if user_id:
            filtered_memories=[m for m in self.memories if m.user_id==user_id]
        else:
            filtered_memories=self.memories
        # 最小重要性得分过滤
        min_importance=kwargs.get("min_importance")
        if min_importance is not None:
            if min_importance<0 or min_importance>1:
                raise ValueError("最小重要性阈值必须在0到1之间")
            filtered_memories=[m for m in filtered_memories if m.importance>=min_importance] 
        # 按时间范围过滤
        time_range=kwargs.get("time_range")
        if time_range is not None:
            if len(time_range)!=2:
                raise ValueError("时间范围必须仅包含开始时间和结束时间")
            start_time=time_range[0]
            end_time=time_range[1]
            if start_time>end_time:
                raise ValueError("开始时间应小于等于结束时间")
            filtered_memories=[m for m in filtered_memories if m.timestamp>=start_time and m.timestamp<=end_time]
        if not filtered_memories:
            return []
        flag=True
        # 语义向量检索
        vector_scores={}
        try:
            # 准备文档
            documents=[query]+[m.content for m in filtered_memories]
            documents=self._translate_to_english(documents)
            # TF-IDF向量化
            vectorizer=TfidfVectorizer(stop_words=None,lowercase=True) # 文本向量化器 不删除停用词(比如中文:了、的、地) 大写转小写
            # 1.学习统一词表 2.将每段文本转化为TF—IDF向量
            tfidf_matrix=vectorizer.fit_transform(documents)
            # 计算相似度
            query_vector=tfidf_matrix[0:1] # 查询向量
            doc_vector=tfidf_matrix[1:] # 回答向量
            # 计算查询向量与每条回答向量的相似度 并展开为一维Numpy数组
            similarities=cosine_similarity(query_vector,doc_vector).flatten()
            # 存储向量分数
            for i,memory in enumerate(filtered_memories):
                vector_scores[memory.id]=similarities[i]
        except Exception  as e:
            flag=False
            vector_scores={}
            print(f"向量检索失败,失败原因为:{e}")
        # 计算最终分数
        query_lower=query.lower()
        scored_memories=[]
        for memory in filtered_memories:
            content_lower=memory.content.lower()
            # 获取向量分数
            vector_score=vector_scores.get(memory.id,0.0)
            # 关键词匹配分数
            keyword_score=0.0
            if query_lower in content_lower:
                keyword_score=self._count_tokens(query_lower)/self._count_tokens(content_lower)
            else:
                # 分词匹配
                query_words=set(self.encoding.encode(query_lower))
                content_words=set(self.encoding.encode(content_lower))
                # 取交集
                intersection=query_words.intersection(content_words)
                # 计算分词匹配得分
                if intersection:
                    keyword_score=len(intersection)/len(query_words.union(content_words))*0.8
            # 混合分数
            if flag:
                base_relevance=vector_score*0.7+keyword_score*0.3
            else:
                base_relevance=keyword_score
            # 时间衰减
            time_decay=self._calculate_time_decay(memory.timestamp)
            base_relevance*=time_decay
            # 重要性权重
            importance_weight=0.8+(memory.importance*0.4)
            final_score=base_relevance*importance_weight
            if final_score>0:
                scored_memories.append((final_score,memory))
        # 按分数排序 降序排列
        scored_memories.sort(key=lambda x:x[0],reverse=True)
        return [memory for _,memory in scored_memories[:limit]]

    # 检查记忆是否存在
    def has_memory(self,memory_id:str):
        # 过期清理
        self._expire_old_memories()
        return any(memory.id==memory_id for memory in self.memories)

    # 清空记忆
    def clear(self):
        self.memories.clear()
        self.memory_heap.clear()
        self.current_tokens=0

    # 获取统计信息
    def get_stats(self):
        # 过期清理
        self._expire_old_memories()
        active_memories=self.memories
        return{
            "count":len(active_memories), # 记忆数量
            "current_tokens":self.current_tokens, # 当前所用token数量
            "max_capacity":self.max_capacity, # 最大容量
            "max_tokens":self.max_tokens, # 工作记忆所能容纳最大token数量
            "max_minutes":self.max_minutes, # TTL
            "session_duration_minutes":(datetime.now()-self.session_start).total_seconds()/60, # 会话记忆存在时长
            "avg_importance":sum(m.importance for m in active_memories)/len(active_memories) if len(active_memories) else 0.0, # 所有记忆平均得分
            "capacity_usage":len(active_memories)/self.max_capacity, # 所用内存比例
            "token_usage":self.current_tokens/self.max_tokens, # 所用token比
            "memory_type":"working" # 记忆系统类型
        }

    # 获取最近若干条记忆
    def get_recent(self,limit:int=10):
        # 过期清理
        self._expire_old_memories()
        # 降序
        sorted_memories=sorted(
            self.memories,
            key=lambda x:x.timestamp,
            reverse=True
        )
        return sorted_memories[:limit]

    # 获取所有记忆
    def get_all(self):
        # 过期清理
        self._expire_old_memories()
        return self.memories.copy()

    # 获取上下文摘要
    def get_context_summary(self,max_length:int=500):
        # 过期清理
        self._expire_old_memories()
        if not self.memories:
            return "No working memories"
        # 按重要性和时间排序
        sorted_memories=sorted(
            self.memories,
            key=lambda m:(m.importance,m.timestamp),
            reverse=True
        )
        summary_parts=[]
        current_length=0
        for memory in sorted_memories:
            content=memory.content
            if current_length+len(content)<=max_length:
                summary_parts.append(content)
                current_length+=len(content)
            else:
                # 截取最后一个记忆
                remaining=max_length-current_length
                # 保留至少50个字符
                if remaining>50:
                    summary_parts.append(content[:remaining]+"...")
                break
        return "Working Memory Context:\n"+"\n".join(summary_parts)

    # 遗忘机制
    def forget(self,strategy:str="importance_based",threshold:float=0.1,max_age_days:int=30):
        forgotton_count=0
        current_time=datetime.now()
        to_remove=[]
        # TTL过期 任何遗忘策略
        cutoff_ttl=current_time-timedelta(minutes=self.max_minutes)
        for memory in self.memories:
            if memory.timestamp<cutoff_ttl:
                to_remove.append(memory.id)
        # 重要性阈值遗忘策略
        if strategy=="importance_based":
            # 删除低重要性记忆
            for memory in self.memories:
                if memory.importance<threshold:
                    to_remove.append(memory.id)
        # 删除过期记忆(与TTL不同)
        elif strategy=="time_based":
            cutoff_time=current_time-timedelta(hours=max_age_days*24)
            for memory in self.memories:
                if memory.timestamp<cutoff_time:
                    to_remove.append(memory.id)
        # 删除超出容量的记忆(add时已检查实际不会触发)
        elif strategy=="capacity_based":
            if len(self.memories)>self.max_capacity:
                sorted_memories=sorted(
                    self.memories,
                    key=lambda m:self._calculate_priority(m)
                )
                excess_count=len(self.memories)-self.max_capacity
                for memory in sorted_memories[:excess_count]:
                    to_remove.append(memory.id)
        to_remove=set(to_remove)
        # 删除
        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotton_count+=1
        return forgotton_count





                
        

            
