from dataclasses import dataclass
from typing import Dict,List,Any,Optional
import hashlib
from datetime import datetime

@dataclass
# 自动生成数据类的常用方法,比如__init__  __repr__
class Document:
    # 文档类
    content:str
    metadata:Dict[str,Any]
    doc_id:Optional[str]=None
    # @dataclass在调用完__init__后会自动调用__post_init__
    def __post_init__(self):
        if self.doc_id is None:
            # 基于内容生成ID
            # string.encode() 将字符串转换为字节数据(默认使用UTF-8)
            # md5()将字符数据转化为MD5 内容相同则结果相同
            # hexdigest() 将MD5结果转化为32位的16进制字符串(原本128位二进制 每4位看成一位)
            self.doc_id=hashlib.md5(self.content.encode()).hexdigest()

@dataclass
class DocumentChunk:
    # 文档块类
    content:str
    metadata:Dict[str,Any]
    chunk_id:Optional[str]=None
    doc_id:Optional[str]=None
    chunk_index:int=0 # 块索引

    def __post_init__(self):
        if self.chunk_id is None:
            # 基于文档ID和块索引生成ID
            chunk_content=f"{self.doc_id}_{self.chunk_index}_{self.content[:50]}"
            self.chunk_id=hashlib.md5(chunk_content.encode()).hexdigest()

# 文档处理器
class DocumentProcessor:
    
    def __init__(self,chunk_size:int=1000,chunk_overlap:int=200,separators:Optional[List[str]]=None):
        self.chunk_size=chunk_size # 文档快最大长度
        self.chunk_overlap=chunk_overlap # 相邻文档快之间重复保留的字符数
        self.separators=separators or ["\n\n","\n","。","."," "] # 切分文档时的分隔符
        # overlap应小于size 下一块起点=当前块起点+size-overlap
    
     # 加载文本文件为文档
    def load_text_file(self,file_path:str,encoding:str="utf-8"):
        with open(file_path,'r',encoding=encoding) as f:
            content=f.read()
        metadata={
            "source":file_path,
            "type":"text_file",
            "loaded_at":datetime.now().isoformat()
        }
        return Document(content=content,metadata=metadata)

    # 将文档分隔成文档块
    def process_document(self,document:Document):
        chunks=self._split_text(document.content)
        document_chunks=[]
        for i,chunk_content in enumerate(chunks):
            # 创建块的元数据
            chunk_metadata=document.metadata.copy()
            chunk_metadata.update({
                "doc_id":document.doc_id,
                "chunk_index":i,
                "total_chunks":len(chunks),
                "processed_at":datetime.now().isoformat()
            })
            # 创建文档快
            chunk=DocumentChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                doc_id=document.doc_id,
                chunk_index=i
            )
            document_chunks.append(chunk)
        return document_chunks

    # 文档切割策略
    def _split_text(self,text:str):
        # text长度小于等于文档块最大长度
        if len(text)<=self.chunk_size:
            return [text]
        chunks=[]
        start=0 # 当前所在位置
        while start<len(text):
            end=start+self.chunk_size
            # 最后一块
            if end>=len(text):
                chunks.append(text[start:])
                break
            # 寻找合适切割点 实际上是下一个开始点
            split_point=self._find_split(text,start,end)
            # 未找到合适切割点 强制分隔
            if split_point==-1:
                split_point=end
            chunks.append(text[start:split_point])
            start=max(start+1,split_point-self.chunk_overlap)
        return chunks

    # 寻找切割点
    def _find_split(self,text:str,start:int,end:int):
        # 按照\n\n \n 。. 以及" "的优先级,即分段>回车>中文句号>英文句号>空格
        for separator in self.separators:
            # 在预计切分位置end前100个字符内寻找合适的分隔符 尽量避免从句子中间强制切断
            search_start=max(start,end-100)
            # 从后向前查找 end-len(separator)~search_start 防止超出end
            for i in range(end-len(separator),search_start-1,-1):
                if text[i:i+len(separator)]==separator:
                    return i+len(separator) # 下一个开始点
        return -1

    # 批量处理文档
    def process_documents(self,documents:List[Document]):
        all_chunks=[]
        for document in documents:
            chunks=self.process_document(document=document)
            all_chunks.append(chunks)
        return all_chunks

    # 合并小的文档快
    def merge_chunks(self,chunks:List[DocumentChunk],max_length:int=2000):
        if not chunks:
            return []
        merged_chunks=[]
        current_chunk=chunks[0]
        for next_chunk in chunks[1:]:
            # 检查是否可以合并
            # 长度不超且是同一文档的文档块
            combined_length=len(current_chunk.content)+len(next_chunk.content)+1
            if combined_length<=max_length and current_chunk.doc_id==next_chunk.doc_id:
                # 合并
                current_chunk.content+="\n"+next_chunk.content
            else:
                merged_chunks.append(current_chunk)
                current_chunk=next_chunk
        # 添加最后一个块
        merged_chunks.append(current_chunk)
        for i,chunk in enumerate(merged_chunks):
            chunk.chunk_index=i
            chunk_content=f"{chunk.doc_id}_{chunk.chunk_index}_{chunk.content[:50]}"
            chunk.chunk_id=hashlib.md5(chunk_content.encode()).hexdigest()
            chunk.metadata.update({
                "chunk_index":i,
                "total_chunks":len(merged_chunks),
                "processed_at":datetime.now().isoformat()
            })
        return merged_chunks

    # 过滤太短的文档块
    def filter_chunks(self,chunks:List[DocumentChunk],min_length:int=50):
        return [chunk for chunk in chunks if len(chunk.content)>=min_length]

    # 为文档块添加元数据
    def add_chunk_metadata(self,chunks:List[DocumentChunk],metadata:Dict[str,Any]):
        for chunk in chunks:
            chunk.metadata.update(metadata)
        return chunks
