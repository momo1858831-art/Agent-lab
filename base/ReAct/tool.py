from serpapi import SerpApiClient
import os
from typing import Any,Dict
from dotenv import load_dotenv

load_dotenv()

def search(query:str):
    # 基于serpapi的网页搜索引擎工具
    print(f"正在使用SerpApi进行网页搜索:{query}")
    try:
        apikey=os.getenv("serpapi_key")
        if not apikey:
            return "serpapi_key未配置"
        params={
            "engine":"google", # 搜索引擎
            "q":query, # 搜索问题
            "api_key":apikey,
            "gl":"cn", # 国家代码
            "hl":"zh-cn" # 语言代码
        }
        client=SerpApiClient(params) # 创建客户端
        results=client.get_dict() # 向serpapi发送请求并将结果转换为py字典
        # 智能解析 优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        # 没有直接答案 取自然搜索结果的前三条
        if "organic_results" in results and results["organic_results"]:
            snippets=[
                f"[{i+1}]{res.get('title', '')}\n{res.get('snippet','')}"
                for i,res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        return f"对不起，没有找到关于'{query}'的信息。"
    except Exception as e:
        return f"搜索时发生{e}错误"

# 管理工具，比如搜索、计算、查询数据库等工具
class ToolExecutor:
    def __init__(self):
        self.tools:Dict[str,Dict[str,Any]]={}
    # 注册工具
    def registerTool(self,name:str,description:str,func:callable):
        if name in self.tools:
            print(f"警告,工具'{name}'已存在,其功能将被覆盖")
        self.tools[name]={"description":description,"func":func}
        print(f"工具'{name}'已注册")
    # 根据工具名取出该工具对应的函数
    def getTool(self,name:str):
        # 有则取名返回函数名 无则取{}返回None
        return self.tools.get(name,{}).get("func")
    # 返回工具名及对应的描述
    def getAvailableTools(self):
        return "\n".join(
            [
                f"{name}:{info['description']}"
                # 取出字典中的所有键值对
                for name,info in self.tools.items()
            ]
        )

if __name__=='__main__':
    toolexceutor=ToolExecutor()
    search_description="一个网页搜索引擎,当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具"
    toolexceutor.registerTool("Search",search_description,search)
    print("---可用工具---")
    print(toolexceutor.getAvailableTools())
    tool_name="Search"
    tool_input="配置为13英寸、内存为1TB、运行内存为24GB的银色MacBook air怎么样"
    tool_function=toolexceutor.getTool(tool_name)
    if tool_function:
        observation=tool_function(tool_input)
        print(observation)
    else:
        print(f"未找到名为'{tool_name}'的工具")