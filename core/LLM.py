import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List,Dict

load_dotenv()

class LLM:

    def __init__(self,model:str=None,apikey:str=None,baseurl:str=None,timeout:int=None):
        # 初始化 若未提供则从.env获取
        self.model=model or os.getenv("model")
        self.apikey=apikey or os.getenv("apikey")
        self.baseurl=baseurl or os.getenv("baseurl")
        self.timeout=timeout or int(os.getenv("timeout"))
        # 检查相关信息是否齐全
        if not all([self.model,self.apikey,self.baseurl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义")
        # 定义LLM
        self.client=OpenAI(
            api_key=self.apikey,
            base_url=self.baseurl,
            timeout=self.timeout
        )

    def think(self,messages:List[Dict[str,str]],temperature:float=0,max_tokens:int=10000):
        # 调用LLM进行思考
        # print(f"目前调用的大语言模型为{self.model}:\n")
        try:
            response=self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature, # temperature越小越稳定 越大回答的创意性越高
                max_tokens=max_tokens,
                stream=True
            )
            # 流式响应
            collected_content=[]
            for chunk in response: # 取每一小块片段
                if not chunk.choices: # 若该片段无候选回答直接跳过
                    continue
                # 取第一个候选回答中的内容 否则取空字符串
                content=chunk.choices[0].delta.content or ""
                print(content,end="",flush=True) # 立即打印
                collected_content.append(content)
            # print("\nLLM调用成功!")
            return "".join(collected_content)
        except Exception as e:
            print(f"LLM调用失败,错误原因:{e}")
            return None

if __name__=='__main__':
    try:
        llm=LLM()
        messages=[
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法,用py语言"}
        ]
        print("正在调用LLM")
        answer=llm.think(messages)
        # if answer: 
            # print(answer)
    except ValueError as e:
        print(e)