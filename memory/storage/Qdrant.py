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
        collection_name:str="hello_agents_vectors", # Qdrant拥有的集合名称
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
            collection_name:str="hello_agents_vectors", # Qdrant拥有的集合名称
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

    # 确保集合存在 不存在则创建
    def _ensure_collection(self):
        try:
            # 检查结合是否存在
            collections=self.client.get_collections().collections
            collection_names=[c.name for c in collections]
            if self.collection_name not in collection_names:
                # 创建新集合
                hnsw_cfg=None
                try:
                    # 同时保存hnsw_m hnsw_ef_construct
                    hnsw_cfg=models.HnswConfigDiff(
                        m=self.hnsw_m,
                        ef_construct=self.hnsw_ef_construct
                    )
                except Exception as e:
                    hnsw_cfg=None
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    ),
                    hnsw_config=hnsw_cfg
                )
                logger.info(f"创建Qdrant集合:{self.collection_name}")
            else:
                logger.info(f"使用现有Qdrant集合:{self.collection_name}")
                # 尝试更新HNSW配置
                try:
                    self.client.update_collection(
                        collection_name=self.collection_name,
                        hnsw_config=models.HnswConfigDiff(
                            m=self.hnsw_m,
                            ef_construct=self.hnsw_ef_construct
                        )
                    )
                except Exception as e:
                    logger.debug(f"跳过更新HNSW配置:{e}")
            self._ensure_payload_indexes()
        except Exception as e:
            logger.error(f"集合初始化失败:{e}")
            raise
    
    # 创建payload索引 快速找到符合条件的数据
    def _ensure_payload_indexes(self):
        try:
            index_fields=[
                ("memory_type",models.PayloadSchemaType.KEYWORD),
                ("user_id",models.PayloadSchemaType.KEYWORD),
                ("memory_id",models.PayloadSchemaType.KEYWORD),
                ("timestamp",models.PayloadSchemaType.INTEGER),
                ("modality",models.PayloadSchemaType.KEYWORD),  # 感知记忆模态筛选
                ("source",models.PayloadSchemaType.KEYWORD),
                ("external",models.PayloadSchemaType.BOOL),
                ("namespace",models.PayloadSchemaType.KEYWORD),
                # RAG相关字段索引
                ("is_rag_data",models.PayloadSchemaType.BOOL),
                ("rag_namespace",models.PayloadSchemaType.KEYWORD),
                ("data_source",models.PayloadSchemaType.KEYWORD),
            ]
            for field_name,schema_type in index_fields:
                try:
                    # 为哪个集合的哪个payload字段创建索引
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type # 字段类型
                    )
                except Exception as e:
                    logger.debug(f"索引{field_name}已存在或创建失败:{e}")
        except Exception as e:
            logger.debug(f"创建payload失败:{e}")

    def add_vectors(
            self,
            vectors:List[List[float]], # 向量
            metadata:List[Dict[str,Any]], # 元数据
            ids:Optional[List[str]]=None # ID列表
    ):
        try:
            if not vectors:
                logger.warning("向量列表为空")
                return False
            # 生成ID
            if ids is None:
                ids=[
                    f"vec_{i}{int(datetime.now().timestamp()*1000000)}"
                    for i in range(len(vectors))
                ]
            # 构建点数据
            logger.info(f"[Qdrant] add_vectors start: n_vectors={len(vectors)} n_metadata={len(metadata)} collection={self.collection_name}")
            points=[]
            for i,(vector,meta,point_id) in enumerate(zip(vectors,metadata,ids)):
                # 确保向量是正确维度
                vlen=len(vector)
                if vlen!=self.vector_size:
                    logger.warning(f"向量维度不匹配,期望维度为{self.vector_size}")
                    continue
                # 添加时间戳到元数据
                meta_with_timestamp=meta.copy()
                meta_with_timestamp["timestamp"]=int(datetime.now().timestamp())
                meta_with_timestamp["added_at"]=int(datetime.now().timestamp())
                if "external" in meta_with_timestamp and not isinstance(meta_with_timestamp.get("external"),bool):
                    val=meta_with_timestamp.get("external")
                    meta_with_timestamp["external"]=True if str(val).lower() in {"1","true","yes"} else False
                # 确保点ID是Qdrant接受的类型(无符号整数或uuid)
                safe_id:Any
                if isinstance(point_id,int):
                    safe_id=point_id
                elif isinstance(point_id,str):
                    try:
                        uuid.UUID(point_id)
                        safe_id=point_id
                    except Exception as e:
                        safe_id=str(uuid.uuid4())
                else:
                    safe_id=str(uuid.uuid4())
                point=PointStruct(
                    id=safe_id, # 唯一标识
                    vector=vector, # 用于相似度计算
                    payload=meta_with_timestamp # 元数据
                )
                points.append(point)
                if not points:
                    logger.warning("没有有效的向量点")
                    return False
                # 批量插入
                operation_info=self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                logger.info("[Qdrant] upsert done")
                logger.info(f"成功添加{len(points)}个向量到Qdrant")
                return True
        except Exception as e:
            logger.error(f"添加向量失败:{e}")
            return False


