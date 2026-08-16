from typing import Union,List,Dict,Optional
import torch

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