from typing import Union,List,Dict,Optional
import torch
import os
from dotenv import load_dotenv
import openai
import threading

load_dotenv()

# 嵌入模型基类
class EmbeddingModel:
    # Union表示texts可以是str也可以是List[str]
    def encode(self,texts:Union[str,List[str]]):
        raise NotImplementedError # 表示还没实现,不能继续执行 需要子类实现(也可用abstractmethod)

    @property # 可以以属性的方式调用该函数 self.dimension
    def dimension(self):
        raise NotImplementedError

# 本地Transformer嵌入
class LocalTransformerEmbedding(EmbeddingModel):

    def __init__(self,model_name:str="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name=model_name # st or hf
        self._backend=None # 使用的后端
        self._st_model=None # 保存st模型对象
        self._hf_tokenizer=None # 保存hf分词器
        self._hf_model=None # 保存hf模型对象
        self._dimension=None # 模型输出的向量维度
        self._load_backend()

    # 加载后端
    def _load_backend(self):
        # 优先st(封装好的高级用法)
        try:
            from sentence_transformers import SentenceTransformer
            # 构建模型
            self._st_model=SentenceTransformer(self.model_name)
            # 将测试文本编码为向量
            test_vec=self._st_model.encode("test_text")
            # 向量维度
            self._dimension=len(test_vec)
            # 后端名
            self._backend="st"
            return
        except Exception as e:
            self._st_model=None
            print(f"st加载失败:{e}")
        # 回退hf(更底层)
        try:
            from transformers import AutoTokenizer,AutoModel
            # 加载分词器 将文本转换为token ID
            self._hf_tokenizer=AutoTokenizer.from_pretrained(self.model_name)
            # 加载模型
            self._hf_model=AutoModel.from_pretrained(self.model_name)
            # 无需自动计算梯度
            with torch.no_grad():
                # pt指返回pytorch张量(token ID)
                # padding=True表示当输入多条文本时,将短文本补齐到最长文本长度
                # truncation=True表示当文本超过模型支持的最大长度时,自动截断多余token
                inputs=self._hf_tokenizer("text_test",return_tensors="pt",padding=True,truncation=True)
                # 解包字典inputs(包含token ID等) 计算token之间的关联
                outputs=self._hf_model(**inputs)
                # 将所有token的向量取平均得到句子向量
                # dim=1表示按列求和再取均值 (句子维度,token维度,向量维度)->(句子维度,向量维度)
                test_embedding=outputs.last_hidden_state.mean(dim=1)
                self._dimension=int(test_embedding.shape[1])
            self._backend="hf"
            return
        except Exception as e:
            self._hf_model=None
            self._hf_tokenizer=None
            print(f"hf加载失败:{e}")
        raise RuntimeError("未找到可用本地嵌入后端")

    def encode(self,texts:Union[str,List[str]]):
        # 只有一条待编码文本
        if isinstance(texts,str):
            inputs=[texts]
            single=True
        else:
            inputs=list(texts)
            single=False
        if self._backend=="st":
            vecs=self._st_model.encode(inputs)
            # 转换为py列表
            vecs=[v for v in vecs]
        else:
            # 后续可改为分批次处理 节省空间
            tokenized=self._hf_tokenizer(inputs,return_tensors="pt",padding=True,truncation=True,max_length=512)
            with torch.no_grad():
                outputs=self._hf_model(**tokenized)
                # 获取每个token的上下文向量 (文本维度,token维度,向量维度)
                token_embeddings=outputs.last_hidden_state
                # tokenized["attention_mask"]形状为(文本维度,token维度) 表示某个文本的某个token是否为padding,取值为0/1
                # 升维为(文本维度,token维度,1)
                attention_mask=tokenized["attention_mask"].unsqueeze(-1)
                # 转换为和token_embeddings相同的数据类型 将0/1转化为小数
                attention_mask=attention_mask.to(token_embeddings.dtype)
                # 将padding位置向量清零(广播) (文本维度,token维度,向量维度)
                # 扩展后每一列完全相同 且每一行的值仅有唯一取值(要么全1要么全0) 全1代表为真实token
                # 逐位置乘后真实token向量不变 padding向量变为0
                masked_embeddings=token_embeddings*attention_mask
                # 将有效Token向量求和 (文本维度,向量维度)
                sum_embeddings=masked_embeddings.sum(dim=1)
                # 计算每条文本的有效Token数量 (文本维度,1)
                # clamp(min=1)将所有小于1的值修改为1 防止除0
                token_count=attention_mask.sum(dim=1).clamp(min=1)
                # 求有效Token的平均向量(广播) (文本维度,向量维度)
                embeddings=(sum_embeddings/token_count).cpu().numpy()
            vecs=[v for v in embeddings]
        if single:
            return vecs[0]
        return vecs

    @property
    def dimension(self):
        return int(self._dimension or 0)

class TFIDFEmbedding(EmbeddingModel):

    # TF 词在当前文档中的重要程度
    # IDF 词在全部文档中的稀有程度

    def __init__(self,max_features:int=1000):
        self.max_features=max_features
        self._vectorizer=None # TF-IDF向量器
        self._is_fitted=False # 向量器是否建立词表
        self._dimension=max_features # 向量维度上限
        self._init_vectorizer()

    # 初始化编码器
    def _init_vectorizer(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            # 词表最多保留多少特征词 自动忽略英文停用词,例如the is are a and of
            self._vectorizer=TfidfVectorizer(max_features=self.max_features,stop_words="english")
        except ImportError:
            raise ImportError("请安装scikit-learn")

    # 建立词表并计算IDF
    def fit(self,texts:List[str]):
        self._vectorizer.fit(texts)
        self._is_fitted=True
        # 词表长度 即向量维度
        self._dimension=len(self._vectorizer.get_feature_names_out())

    # 计算TF-IDF向量
    def encode(self,texts:Union[str,List[str]]):
        if not self._is_fitted:
            raise ValueError("TF-IDF模型未训练,请先调用fit方法")
        single=False
        if isinstance(texts,str):
            texts=[texts]
            single=True
        # 稀疏矩阵
        tfidf_matrix=self._vectorizer.transform(texts)
        # 转化为普通数组
        embeddings=tfidf_matrix.toarray()
        if single:
            return embeddings[0]
        return [e for e in embeddings]

    @property
    def dimension(self):
        return self._dimension

# 调用Embedding API
class APIEmbedding(EmbeddingModel):

    def __init__(self,model_name:Optional[str]=None,api_key:Optional[str]=None,base_url:Optional[str]=None):
        self.model_name=model_name or os.getenv("embedding_model")
        self.api_key=api_key or os.getenv("embedding_api")
        self.base_url=base_url or os.getenv("embedding_baseurl")
        self._dimension=None

    def encode(self,texts:Union[str,List[str]]):
        if not self.api_key:
            raise ValueError("API_KEY未配置")
        if not self.base_url:
            raise ValueError("BASE_URL未配置")
        if not self.model_name:
            raise ValueError("MODEL未配置")
        single=False
        if isinstance(texts,str):
            inputs=[texts]
            single=True
        else:
            inputs=texts
        client=openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        response=client.embeddings.create(
            input=inputs,
            model=self.model_name
        )
        vecs=[item.embedding for item in response.data]
        self._dimension=len(vecs[0])
        if single:
            return vecs[0]
        return vecs

    @property
    def dimension(self):
        return self._dimension or 0

_lock=threading.RLock()
_embedder:Optional[EmbeddingModel]=None

# 创建嵌入模型实例
def create_embedding_model(model_type:str="local",**kawrgs):
    if model_type=="local":
        return LocalTransformerEmbedding()
    elif model_type=="dashscope":
        return APIEmbedding(**kawrgs)
    elif model_type=="tfidf":
        return TFIDFEmbedding()
    else:
        raise ValueError(f"不支持的模型:{model_type}")

# 带回退的创建
def create_embedding_model_with_fallback(preferred_type:str="dashscope",**kawrgs):
    if preferred_type in ("sentence_transformer","huggingface"):
        preferred_type="local"
    fallback=["dashscope","local","tfidf"]
    # 将首选放前面
    if preferred_type in fallback:
        fallback.remove(preferred_type)
        fallback.insert(0,preferred_type)
    for t in fallback:
        try:
            return create_embedding_model(t,**kawrgs)
        except Exception:
            continue
    raise RuntimeError("所有嵌入模型均不可用")
    

def _build_embedder():
    preferred=os.getenv("embedding_type","dashscope").strip()
    # 根据提供商选择默认模型
    model_name="text-embedding-v4" if preferred=="dashscope" else "sentence-transformers/all-MiniLM-L6-v2"
    kawrgs={}
    kawrgs["model_name"]=model_name
    api_key=os.getenv("embedding_api")
    base_url=os.getenv("embedding_baseurl")
    if api_key:
        kawrgs["api_key"]=api_key
    if base_url:
        kawrgs["base_url"]=base_url
    return create_embedding_model_with_fallback(preferred_type=preferred,**kawrgs)

# 获取全局共享的文本嵌入实例
def get_text_embedder():
    # 声明全局变量
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            _embedder=_build_embedder()
        return _embedder

# 获取统一向量维度
def get_dimension():
    embedder=get_text_embedder()
    dimension=embedder.dimension
    if dimension is None or dimension==0 or (isinstance(embedder,TFIDFEmbedding) and not embedder._is_fitted):
        raise ValueError("请先确定embedding维度")
    return dimension
    
    
    
