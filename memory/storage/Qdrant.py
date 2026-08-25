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
        # 搜索过程中,保留与查询向量最接近的search_ef个候选向量。
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
        self.distance=distance_map.get(distance.lower(),Distance.COSINE)
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
                        # 验证point_id是否为uuid格式的字符串 但不会改变point_id
                        uuid.UUID(point_id)
                        safe_id=point_id
                    except Exception as e:
                        safe_id=str(uuid.uuid4())
                else:
                    safe_id=str(uuid.uuid4())
                point=PointStruct(
                    id=safe_id, # 唯一标识 无符号整数或uuid格式的字符串
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

    # 搜索相似向量
    def search_similar(
            self,
            query_vector:List[float], # 查询向量
            limit:int=10, # 返回结果数量限制
            score_threshold:Optional[float]=None, # 相似度阈值
            where:Optional[Dict[str,Any]]=None # 过滤条件
    ):
        try:
            if len(query_vector)!=self.vector_size:
                logger.error(f"查询向量维度错误,期望维度为{self.vector_size}")
                return []
            # 构建过滤器
            query_filter=None
            if where:
                conditions=[]
                for key,value in where.items():
                    if isinstance(value,(str,int,float,bool)):
                        # 检查payload字段
                        # payload[key]=value
                        conditions.append(
                            FieldCondition(
                                key=key, # 指定字段名称
                                # 精准匹配
                                match=MatchValue(value=value) # 字段值是否等于value
                            )
                        )
                if conditions:
                    # must表示所有条件均需满足
                    query_filter=Filter(must=conditions)
            # 搜索参数
            search_params=None
            # 构造搜索配置对象
            try:
                search_params=models.SearchParams(
                    hnsw_ef=self.search_ef, # 搜索范围
                    exact=self.search_exact # 精确搜索(符合过滤条件的所有向量)
                )
            except Exception as e:
                search_params=None
            # 执行搜索
            search_results=self.client.query_points(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True, # 返回结果包含payload字段
                with_vectors=False, # 不返回原始向量
                search_params=search_params
            )
            # 转换结果格式 原先为ScorePoint对象,包含id score payload vector等
            results=[]
            for hit in search_results:
                result={
                    "id":hit.id, # 插入数据时该数据的ID
                    "score":hit.score, # 相似度分数
                    "metadata":hit.payload or {} # 元数据
                }
                results.append(result)
            logger.debug(f"Qdrant搜索返回{len(results)}条结果")
            return results
        except Exception as e:
            logger.error(f"向量搜索失败:{e}")
            return []

    def delete_vectors(self,ids:List[str]):
        try:
            if not ids:
                return True
            operation_info=self.client.delete(
                collection_name=self.collection_name,
                # 数据点ID选择器 设置删除点
                # PointIdsList按ID删除
                points_selector=models.PointIdsList(
                    points=ids
                ),
                wait=True # 等待删除真正完成后再返回
            )
            logger.info(f"成功删除{len(ids)}个向量")
            return True
        except Exception as e:
            logger.error(f"删除向量失败:{e}")
            return False

    # 清空集合
    def clear_collection(self):
        try:
            # 删除集合
            self.client.delete_collection(collection_name=self.collection_name)
            # 重新创建集合
            self._ensure_collection()
            logger.info(f"成功清空Qdrant集合:{self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"清空集合失败:{e}")
            return False

    # 删除指定记忆(为后续记忆系统类型扩展做铺垫)
    def delete_memories(self,memory_ids:List[str]):
        try:
            if not memory_ids:
                return
            # payload[memory_id]=mid
            # 按ID过滤用HasIdCondition
            conditions=[
                FieldCondition(
                    key="memory_id",
                    match=MatchValue(value=mid)
                )
                for mid in memory_ids
            ]
            query_filter=Filter(should=conditions) # 满足一个条件即可
            self.client.delete(
                collection_name=self.collection_name,
                # FilterSelector按payload字段删除
                points_selector=models.FilterSelector(
                    filter=query_filter
                ),
                wait=True
            )
            logger.info(f"成功按memory_id删除{len(memory_ids)}条记忆")
        except Exception as e:
            logger.error(f"删除记忆失败:{e}")
            raise

    # 获取集合信息
    def get_collection_info(self):
        try:
            # 获取指定集合的基本信息
            collection_info=self.client.get_collection(self.collection_name)
            info={
                "name":self.collection_name,
                # 已经建立专用向量索引的向量数
                "indexed_vectors_count":collection_info.indexed_vectors_count,
                # 数据点数量(一个数据点可能包含多个向量 此处没用)
                "points_count":collection_info.points_count,
                # 每个集合可能划分为多个Segment 提高效率,如并行搜索 避免每次数据变化都操作整个集合
                "segments_count":collection_info.segments_count,
                "config":{
                    "vector_size":self.vector_size,
                    "distance":self.distance.value,
                }
            }
            return info
        except Exception as e:
            logger.error(f"获取集合信息失败:{e}")
            return {}

    def get_collection_stats(self):
        info=self.get_collection_info()
        if not info:
            return {"store_type":"qdrant","name":self.collection_name}
        info["store_type"]="qdrant"
        return info

    # 健康检查
    def health_check(self):
        try:
            collections=self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant健康检查失败:{e}")
            return False

    # 析构函数 关闭当前QdrantClient 不会删除Qdrant的向量
    # 复用该QdrantClient的调用者也会收到影响 适合整个应用退出时调用
    def __del__(self):
        if hasattr(self,'client') and self.client:
            try:
                self.client.close()
            except:
                pass
            
        

