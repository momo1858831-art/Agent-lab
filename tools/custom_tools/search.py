import os
import serpapi
from typing import Any,Dict
from tavily import TavilyClient
from typing import Optional
from ..tool import Tool
from ..tool_parameter import ToolParameter
from ..tool_register import Toolregistr
from dotenv import load_dotenv

load_dotenv()

class SearchTool(Tool):

    # 三种搜索模数
    # 1.混合模式hybrid 
    # 2.tavily AI搜索
    # 3.serpapi Google搜索
    def __init__(self,backend:str="hybrid",tavily_key:Optional[str]=None,serpapi_key:Optional[str]=None):
        super().__init__(
            name="search",
            description="一个智能网页搜索引擎 支持混合搜索模式 自动选择最佳搜索源"
        )
        self.backend=backend
        self.tavily_key=tavily_key or os.getenv("tavily_key")
        self.serpapi_key=serpapi_key or os.getenv("serpapi_key")
        self.available_backends=[]
        self._setup_backends()

    def get_parameters(self):
        return [
            ToolParameter(
                name="query",
                type="string",
                description="需要搜索的关键词或问题",
                required=True
            )
        ]

    def _setup_backends(self):
        if self.tavily_key:
            self.available_backends.append("tavily")
        if self.serpapi_key:
            self.available_backends.append("serpapi")

    def search_hybrid(self,query:str):
        # 优先tavily
        if "tavily" in self.available_backends:
            try:
                return self.search_tavily(query)
            except Exception as e:
                print(f"Tavily调用失败,失败原因为:{e}")
                if "serpapi" in self.available_backends:
                    print("正在切换至serpapi搜索:")
                    try:
                        return self.search_serpapi(query)
                    except Exception as e:
                        print(f"serpapi调用失败,失败原因为:{e}")
        elif "serpapi" in self.available_backends:
            try:
                return self.search_serpapi(query)
            except Exception as e:
                print(f"serpapi调用失败,失败原因为:{e}")
        else:
            return "无可用搜索源"
        return "搜索源均调用失败"

    def search_tavily(self,query:str):
        client=TavilyClient(api_key=self.tavily_key)
        response=client.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=3
        )
        result=f"Tavily搜索结果为:\n{response.get('answer','')}\n"
        for i,item in enumerate(response.get('results',[])[:3]):
            result+=f"[{i+1}] {item.get('title')}\n"
            result+=f"{item.get('content','')[:200]}...\n"
            result+=f"来源:{item.get('url','')}\n"
        return result

    def search_serpapi(self,query:str):
        client=serpapi.Client(api_key=self.serpapi_key)
        response=client.search(
            {
                "engine":"google", 
                "q":query, 
                "gl":"cn", 
                "hl":"zh-cn" 
            }
        )
        results="Serpapi搜索结果为:\n"
        for i,item in enumerate(response.get('organic_results',[])[:3]):
            results+=f"[{i+1}] {item.get('title','')}\n"
            results+=f"{item.get('snippet','')}\n"
            results+=f"来源:{item.get('link','')}\n"
        return results

    def get_parameters(self):
        return [
            ToolParameter(
                name="query",
                type="string",
                description="需要搜索的关键词或问题",
            )
        ]

    def run(self,parameters:Dict[str,Any]):
        query=parameters.get("query")
        if not query:
            return "缺少必填参数query"
        if self.backend=="hybrid":
            return self.search_hybrid(query)
        if self.backend=="tavily":
            if "tavily" not in self.available_backends:
                return "Tavily未配置API Key"
            return self.search_tavily(query)
        if self.backend=="serpapi":
            if "serpapi" not in self.available_backends:
                return "Serpapi未配置API Key"
            return self.search_serpapi(query)
        print(self.available_backends)
        return "搜索源均不可用"

if __name__=='__main__':
    test_query = "py虚拟环境是什么"
    backend="serpapi"
    search_tool=SearchTool(backend=backend)
    register=Toolregistr()
    register.register_tool(search_tool)
    if backend not in search_tool.available_backends:
        print(f"无{backend}对应的相关配置")
    else:
        print(search_tool.run({"query":test_query}))
    

