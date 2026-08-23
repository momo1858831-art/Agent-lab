import os
from typing import List,Dict,Optional,Any
import hashlib

def _get_markdown_instance():
    try:
        from markitdown import MarkItDown
        # 创建一个文档转换器实例
        return MarkItDown()
    except ImportError:
        print("缺少包:markitdown")
        return None

# 检查是否支持某文件格式
def is_markitdown_supported_format(path:str):
    # [1]获取文件扩展名 [0]获取文件路径
    ext=(os.path.splitext(path)[1] or '').lower()
    supproted_formats={
        # 文档
        '.pdf','.doc','.docx','.xls','.xlsx','.ppt','.pptx',
        # 文本
        '.txt','.md','.csv','.json','.xml','.html','.htm',
        # 图片
        '.jpg','.jpeg','.png','.gif','.bmp','.tiff','.tif','.webp',
        # 音频
        '.mp3','.wav','.m4a','.aac','.flac','.ogg',
        # 压缩包
        '.zip','.tar','.gz','.rar',
        # code
        '.py','.js','.ts','.java','.cpp','.c','ch','.css','.scss',
        # 其他
        '.log','.conf','.ini','.cfg','.yaml','.yml','.toml'
    }
    return ext in supproted_formats

# 将支持的文件转换为md格式的字符串
def _convert_to_markdown(path:str):
    # 判断文件是否存在
    if not os.path.exists(path):
        return ""
    # 对PDF使用增强处理
    ext=(os.path.splitext(path)[1] or '').lower()
    if ext=='.pdf':
        return _enhanced_pdf_processing(path)
    # 其他格式使用原有MarkitDown
    md_instance=_get_markdown_instance()
    if md_instance is None:
        # 直接读取文件内容
        return _fallback_text_reader(path)
    try:
        # 转换结果对象
        result=md_instance.convert(path)
        # md格式的字符串
        text=result.markdown
        if isinstance(text,str) and text.strip():
            return text
    except Exception as e:
        print(f"MarkitDown失败:{e}")
        return _fallback_text_reader(path)

# PDF增强处理
def _enhanced_pdf_processing(path:str):
    # 使用现有MarkitDown提取
    md_instance=_get_markdown_instance()
    if md_instance is None:
        return read_pdf(path)
    try:
        result=md_instance.convert(path)
        raw_text=result.markdown
        if not raw_text or not raw_text.strip():
            return ""
        # 后处理:清理和重组文本
        cleaned_text=_post_process_pdf_text(raw_text)
        print(f"PDF后处理完成")
        return cleaned_text
    except Exception as e:
        print(f"PDF增强处理失败:{e}")
        return read_pdf(path)

# MarkitDown不可用 直接读取文件内容
def _fallback_text_reader(path:str):
    try:
        # 遇到无法解码的字符直接忽略
        with open(path,'r',encoding='utf-8',errors='ignore') as f:
            return f.read()
    except Exception:
        try:
            with open(path,'r',encoding='latin-1',errors='ignore') as f:
                return f.read()
        except Exception:
            return ""

# PDF专用回退
def read_pdf(path:str):
    try:
        import fitz
        pages=[]
        with fitz.open(path) as document:
            for page in document:
                text=page.get_text("text").strip()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        print(f"PDF解析失败:{e}")
        return ""

# 对PDF进行后处理以提高质量
def _post_process_pdf_text(text:str):
    import re # 正则表达式
    # 按行分割处理
    lines=text.splitlines()
    cleaned_lines=[]
    for line in lines:
        line=line.strip()
        if not line:
            continue
        # 移除明显的页眉页脚噪音
        # 以1个或多个数字开头 以1个或多个数字结尾 因此是纯数字
        if re.match(r'^\d+$',line): # 纯数字行(页码)
            continue
        cleaned_lines.append(line)
    # 合并短行
    merged_lines=[]
    i=0
    while i<len(cleaned_lines):
        current_line=cleaned_lines[i]
        # 如果当前行很短 尝试与下一行合并
        if len(current_line)<60 and i+1<len(cleaned_lines):
            next_line=cleaned_lines[i+1]
            # 合并条件:
            # 当前行和下一行都不是标题
            # 当前行不以冒号结尾
            if (not current_line.endswith('：') and (not current_line.endswith(':')) and
            (not current_line.startswith('#')) and (not next_line.startswith('#')) and
            len(next_line)<120):
                merged_line=current_line+" "+next_line
                merged_lines.append(merged_line)
                i+=2 # 跳过下一行
                continue
        merged_lines.append(current_line)
        i+=1
    # 重新组织段落
    paragraphs=[]
    current_paragraph=[]
    for line in merged_lines:
        # 检查是否是新段落的开始(单独段落)
        if (line.startswith('#') or # 标题
        line.endswith(':') or # 英文冒号结尾
        line.endswith('：')): # 中文冒号结尾
            # 保存当前段落
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph=[]
            paragraphs.append(line)
        else:
            current_paragraph.append(line)
    # 添加最后一个段落
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    return '\n\n'.join(paragraphs)

# 检测一段文本使用的主要语言 中文/英文...
def _detect_lang(sample:str):
    try:
        from langdetect import detect
        return detect(sample[:1000]) if sample else "unknown"
    except Exception:
        return "unknown"

# 判断一个字符是否为汉字
def _is_cjk(ch:str):
    if len(ch)!=1:
        return False
    # 将字符转化为Unicode码点整数
    code=ord(ch)
    return (
        # 判断是否为汉字
        0x4E00<=code<=0x9FFF or # 常用汉字
        0x3400<=code<=0x4DBF or # 扩展A区 一血少见汉字
        0x20000<=code<=0x2A6DF or # 扩展B区 生僻字
        0x2A700<=code<=0x2B73F or # ...
        0x2B740<=code<=0x2B81F or
        0x2B820<=code<=0x2CEAF or
        0xF900<=code<=0xFAFF
    )

# 计算一段文本的token数量
def _approx_token_len(text:str):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except ImportError:
        # 中文
        cjk=sum(1 for ch in text if _is_cjk(ch))
        # 英文
        non_cjk_text="".join(" " if _is_cjk(ch) else ch for ch in text)
        non_cjk_tokens=len(non_cjk_text.split())
        return cjk+non_cjk_tokens

# 为md字符串划分段落信息
def _split_paragraphs_with_headings(text:str):
    lines=text.splitlines()
    heading_stack:List[str]=[] # 当前正文的标题路径
    paragraphs:List[Dict]=[] # 保存切分的每个段落的相关内容
    buf:List[str]=[] # 缓冲区
    char_pos=0 # 当前所在位置
    # 将缓冲区的内容添加至段落
    def flush_buf(end_pos:int):
        if not buf:
            return
        content="\n".join(buf).strip()
        if not content:
            return
        paragraphs.append({
            "content":content,
            "heading_path":" > ".join(heading_stack) if heading_stack else None,
            "start":max(0,end_pos-len(content)),
            "end":end_pos
        })
    for line in lines:
        raw=line
        # 标题
        if raw.strip().startswith('#'):
            flush_buf(char_pos)
            # 计算#数量 确定几级标题
            level=len(raw)-len(raw.lstrip('#'))
            # 得到标题正文
            title=raw.lstrip('#').strip()
            # 更新标题路径
            if level<=len(heading_stack):
                heading_stack=heading_stack[:level-1]
            heading_stack.append(title)
            # 更新当前字符位置
            char_pos+=len(raw)+1
            continue
        # 空行表示段落结束
        if raw.strip()=="":
            flush_buf(char_pos)
            buf=[]
        # 普通正文加到缓冲区
        else:
            buf.append(raw)
        char_pos+=len(raw)+1
    flush_buf(char_pos)
    if not paragraphs:
        paragraphs=[{
            "content":text,
            "heading_path":None,
            "start":0,
            "end":len(text)
        }]
    return paragraphs

# 将段落划分为chunk
def _chunk_paragraphs(paragraphs:List[Dict],chunk_tokens:int,overlap_tokens:int):
    chunks:List[Dict]=[]
    cur:List[Dict]=[]
    cur_tokens=0
    i=0
    while i<len(paragraphs):
        p=paragraphs[i]
        p_tokens=_approx_token_len(p["content"])
        # 超长段落单独加入
        if cur_tokens+p_tokens<=chunk_tokens or not cur:
            cur.append(p)
            cur_tokens+=p_tokens
            i+=1
        else:
            content="\n\n".join(x["content"] for x in cur)
            start=cur[0]["start"]
            end=cur[-1]["end"]
            # 取出第一个(逆序)heading_path
            heading_path=next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")),None)
            chunks.append({
                "content":content,
                "start":start,
                "end":end,
                "heading_path":heading_path
            })
            # 选取当前chunk末尾的一些段落作为重叠部分 保证切块间上下文的连续性
            if overlap_tokens>0 and cur:
                kept:List[Dict]=[]
                kept_tokens=0
                # 直接用overlap可能不断重复添加重叠部分
                # 限制重叠内容 为下一个待处理段落预留空间 避免死循环
                max_overlap_tokens=min(overlap_tokens,max(0,chunk_tokens-p_tokens))
                for x in reversed(cur):
                    t=_approx_token_len(x["content"])
                    if kept_tokens+t>max_overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens+=t
                # 重叠部分
                cur=list(reversed(kept))
                cur_tokens=kept_tokens
            # 未启用overlap
            else:
                cur=[]
                cur_tokens=0
    if cur:
        content="\n\n".join(x["content"] for x in cur)
        start=cur[0]["start"]
        end=cur[-1]["end"]
        # 取出第一个(逆序)heading_path
        heading_path=next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")),None)
        chunks.append({
            "content":content,
            "start":start,
            "end":end,
            "heading_path":heading_path
        })
    return chunks

# 将文件划分为chunk
def load_and_chunk_texts(paths:List[str],chunk_size:int=800,chunk_overlap:int=100,namespace:Optional[str]=None,source_label:str="rag"):
    chunks:List[Dict]=[]
    seen_hashes=set()
    # 处理不同文件
    for path in paths:
        if not os.path.exists(path):
            print("File not found")
            continue
        print(f"RAG processing:{path}")
        # 文件扩展名
        ext=(os.path.splitext(path)[1] or '').lower()
        # 转化为md格式字符串
        markdown_text=_convert_to_markdown(path)
        if not markdown_text.strip():
            print(f"warning:未从{path}提取到任何内容")
            continue
        # 文本主要语言
        lang=_detect_lang(markdown_text)
        # 按内容(字符串)生成文档id
        doc_id=hashlib.md5(f"{path}|{len(markdown_text)}".encode('utf-8')).hexdigest()
        # 将文本划分为不同段落
        para=_split_paragraphs_with_headings(markdown_text)
        # 将段落划分为不同chunk
        token_chunks=_chunk_paragraphs(para,chunk_tokens=chunk_size,overlap_tokens=chunk_overlap)
        for ch in token_chunks:
            content=ch["content"]
            start=ch["start"]
            end=ch["end"]
            norm=content.strip()
            if not norm:
                continue
            # 内容id
            content_hash=hashlib.md5(norm.encode('utf-8')).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            # chunk id
            chunk_id=hashlib.md5(f"{doc_id}|{start}|{end}|{content_hash}".encode('utf-8')).hexdigest()
            # 在原先chunk基础上添加相关信息
            chunks.append({
                "id":chunk_id,
                "content":content,
                "metadata":{
                    "source_path":path,
                    "file_ext":ext,
                    "doc_id":doc_id,
                    "lang":lang,
                    "start":start,
                    "end":end,
                    "content_hash":content_hash,
                    "namespace":namespace or "default", # 数据所属知识库
                    "source":source_label, # 数据来源标签
                    "external":True, # 是否为外部导入的数据
                    "heading_path":ch.get("heading_path"),
                    "format":"markdown", # 标准化格式
                },
            })
    print(f"RAG加载完成,共{len(chunks)}条记录切片")
    return chunks


    


