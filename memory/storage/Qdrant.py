import logging
import os
import uuid
import threading
from typing import Dict, List, Optional, Any, Union
import numpy as np
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import(
        Distance,VectorParams,PointStruct, 
        Filter,FieldCondition,MatchValue,SearchRequest
    )
    QDRANT_AVAILABLE=True

except ImportError:
    QDRANT_AVAILABLE=False
    QdrantClient=None
    models=None

logger=logging.getLogger(__name__)

# Qdrant链接管理器
class QdrantConnectionManager:

    _instances={}
    _lock=threading.Lock() # 线程互斥锁

    @classmethod
    def get_instance(
        cls,
        url:Optional[str]=None, # Qdrant服务器地址
        api_key:Optional[str]=None, # Qdrant服务器认证密钥
        collection_name:str="hello_agents_vectors", # 要使用的Qdrant集合名称
        vector_size:int=384, # 向量维度
        distance:str="cosine", # 计算向量相似度的方法
        timeout:int=30,
        **kwargs
    ):
        key=(url or "local",collection_name)
        if key not in cls._instances:
            with cls._lock:
                # 双重检查锁定
                # 防止同一进程的不同线程重复创建同一Qdrant连接
                if key not in cls._instances:
                    logger.debug(f"创建新的Qdrant连接:{collection_name}")
                    cls._instances[key]=QdrantVectorStore(
                        url=url,
                        api_key=api_key,
                        collection_name=collection_name,
                        vector_size=vector_size,
                        distance=distance,
                        timeout=timeout,
                        **kwargs
                    )
                else:
                    logger.debug(f"复用现有Qdrant连接:{collection_name}")
        else:
            logger.debug(f"复用现有Qdrant连接:{collection_name}")
        return cls._instances[key]

# Qdrant向量数据库
class QdrantVectorStore:

    def __init__(
            self,
            url:Optional[str]=None, # Qdrant服务器地址
            api_key:Optional[str]=None, # Qdrant服务器认证密钥
            collection_name:str="hello_agents_vectors", # 要使用的Qdrant集合名称
            vector_size:int=384, # 向量维度
            distance:str="cosine", # 计算向量相似度的方法
            timeout:int=30,
            **kwargs
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError("未安装Qdrant")
        self.url=url # 如果为None则代表本地
        self.api_key=api_key or os.getenv("Qdrant_apikey")
        self.collection_name=collection_name
        self.vector_size=vector_size
        self.timeout=timeout
        # 每个向量节点最多保留多少个临近向量的连接
        try:
            self.hnsw_m=int(os.getenv("QDRANT_HNSW_M",32))
        except Exception:
            self.hnsw_m=32
        # 每个向量插入时选取邻居向量的搜索范围
        try:
            self.hnsw_ef_construct=int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT",256))
        except Exception:
            self.hnsw_ef_construct=256
        # 查询时的搜索范围
        try:
            self.search_ef=int(os.getenv("QDRANT_SEARCH_EF","128"))
        except Exception:
            self.search_ef=128
        # 是否启用精确向量搜索
        self.search_exact=os.getenv("QDRANT_SEARCH_EXACT","0")=="1"
        # 距离向量映射
        distance_map={
            "cosine":Distance.COSINE, # 余弦相似度
            "dot":Distance.DOT, # 点积
            "euclidean":Distance.EUCLID # 欧式距离
        }
        self.distance=distance_map.get(distance.lower(),Distance.COSINE)\
        # 初始化客户端
        self.client=None
        self._initialize_client()

    # 初始化客户端和集合
    def _initialize_client(self):
        try:
            # 根据配置创建客户端连接
            if self.url and self.api_key:
                # 云服务API
                self.client=QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
                logger.info(f"成功连接到Qdrant:{self.url}")
            else:
                # 使用本地服务
                self.client=QdrantClient(
                    host="localhost",
                    port=6333,
                    timeout=self.timeout
                )
                logger.info(f"成功连接到本地Qdrant:localhost:6333")
            # 检查连接
            collections=self.client.get_collections()
            # 创建或获取集合
            self._ensure_collection()
        except Exception as e:
            logger.info(f"Qdrant连接失败:{e}")
            raise

    
