import os
from typing import Optional
from openai import OpenAI
from LLM import LLM



class LLMExtension(LLM):

    def set_provider(self,model,apikey,baseurl,provider,**kwargs):
        if provider=="openai":
            print("正在使用openai下的大模型")
            model = model or os.getenv("openai_model")
            apikey=apikey or os.getenv("openai_apikey")
            baseurl=baseurl or os.getenv("openai_baseurl")
            temperature=kwargs.get("temperature",float(os.getenv("openai_temperature",0)))
            max_tokens=kwargs.get("max_tokens",int(os.getenv("openai_max_tokens",10000)))
            timeout=kwargs.get("timeout",int(os.getenv("openai_timeout",60)))
        elif provider=="deepseek":
            print("正在使用deepseek下的大模型")
            model = model or os.getenv("deepseek_model")
            apikey=apikey or os.getenv("deepseek_apikey")
            baseurl=baseurl or os.getenv("deepseek_baseurl")
            temperature=kwargs.get("temperature",float(os.getenv("deepseek_temperature",0)))
            max_tokens=kwargs.get("max_tokens",int(os.getenv("deepseek_max_tokens",10000)))
            timeout=kwargs.get("timeout",int(os.getenv("deepseek_timeout",60)))
        else:
            raise ValueError("暂不支持此供应商")
        return model,apikey,baseurl,temperature,max_tokens,timeout

    # 支持多提供商
    def __init__(self,model:Optional[str]=None,apikey:Optional[str]=None,baseurl:Optional[str]=None,provider:Optional[str]="auto", **kwargs):
        # 检查提供商是否由用户指定
        if provider!="auto":
            self.provider=provider
            self.model,self.apikey,self.baseurl,self.temperature,self.max_tokens,self.timeout=self.set_provider(
                model,
                apikey,
                baseurl,
                provider,
                **kwargs
            )
            self.client=OpenAI(
                api_key=self.apikey,
                base_url=self.baseurl,
                timeout=self.timeout
            )
        else:
            super().__init__(
                model=model,
                apikey=apikey,
                baseurl=baseurl,
                timeout=kwargs.get('timeout',60)
            )

if __name__=='__main__':
    llmex=LLMExtension(provider="deepseek")
    messages=[
        {
            "role":"user",
            "content":"1+2=?"
        }
    ]
    llmex.think(messages)

        

