import concurrent.futures
import asyncio
from typing import Dict,List
from .tool_register import Toolregistr
from .tool import Tool
from .custom_tools.search import SearchTool

class AsyncToolExecutor:

    def __init__(self,registry:Toolregistr,max_workers:int=4):
        self.registry=registry
        # 创建一个线性池
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def executor_tool_async(self,tool_name:str,input_data:str):
        # 当前正在工作的异步任务调度器
        loop=asyncio.get_running_loop()
        def _execute():
            return self.registry.execute_tool(tool_name,input_data)
        # 将任务_execute交给线程池self.executor 等待执行完成
        result=await loop.run_in_executor(self.executor,_execute)
        return result

    async def execute_tools_parallel(self,tasks:List[Dict[str,str]]):
        # 并行执行多个工具任务
        print(f"开始并行执行{len(tasks)}个工具")
        async_tasks=[]
        for task in tasks:
            tool_name=task["tool_name"]
            input_data=task["input_data"]
            # py并不会立刻执行完整函数 先返回协程对象
            async_task=self.executor_tool_async(tool_name,input_data)
            async_tasks.append(async_task)
        # 同时执行async_tasks中的所有任务 并等待它们在线程池全部完成
        results=await asyncio.gather(*async_tasks)
        print("所有工具任务执行完成")
        return results

    def __del__(self):
        # 检查当前对象是否有executor属性
        if hasattr(self,'executor'):
            # 关闭线性池 wait=True表示等待已经提交的任务全部执行完
            self.executor.shutdown(wait=True)

if __name__=="__main__":
    tool=SearchTool()
    tool_register=Toolregistr()
    tool_register.register_tool(tool)
    executor=AsyncToolExecutor(tool_register)
    tasks=[
        {
            "tool_name":"search",
            "input_data":'{"query": "什么是快乐星球"}'
        },
        {
            "tool_name":"search",
            "input_data":'{"query": "谁是世界上最帅的男人"}'
        }
    ]
    results=asyncio.run(executor.execute_tools_parallel(tasks))
    for result in results:
        print(result)

