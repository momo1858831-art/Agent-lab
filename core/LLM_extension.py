import os
from typing import Optional
from openai import OpenAI
from core.LLM import LLM



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
        elif provider=="Anthropic":
            print("正在使用Anthropic下的大模型")
            model = model or os.getenv("Anthropic_model")
            apikey=apikey or os.getenv("Anthropic_apikey")
            baseurl=baseurl or os.getenv("Anthropic_baseurl")
            temperature=kwargs.get("temperature",float(os.getenv("Anthropic_temperature",0)))
            max_tokens=kwargs.get("max_tokens",int(os.getenv("Anthropic_max_tokens",10000)))
            timeout=kwargs.get("timeout",int(os.getenv("Anthropic_timeout",60)))
        else:
            raise ValueError("暂不支持此供应商")
        return model,apikey,baseurl,temperature,max_tokens,timeout

    # 支持多提供商
    def __init__(self,model:Optional[str]=None,apikey:Optional[str]=None,baseurl:Optional[str]=None,provider:Optional[str]="auto", **kwargs):
        self.provider=provider
        # 检查提供商是否由用户指定
        if provider!="auto":
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


# 自动检测提供商
def auto_detect_provider(apikey:Optional[str]=None,baseurl:Optional[str]=None):
    result=[]
    # 环境变量
    if os.getenv("openai_apikey"):
       result.append("openai")
    if os.getenv("deepseek_apikey"):
        result.append("deepseek")
    if os.getenv("Anthropic_apikey"):
        result.append("Anthropic")
    # 获取环境变量
    api_key=apikey
    base_url=baseurl
    # 根据base_url判断
    if base_url:
        base_url_lowwer=base_url.lower()
        if "openai" in base_url_lowwer:
            result.append("openai")
        if "deepseek" in base_url_lowwer:
            result.append("deepseek")
        if "anthropic" in base_url_lowwer:
            result.append("Anthropic")
    # 也可根据不同模型的密钥格式进行判断
    # if api_key:
    result.append("auto")
    result=list(set(result))
    return result

# 根据提供商解析密钥和服务商地址
def auto_resolve_credentials(provider):
    if provider=="openai":
        api_key=os.getenv("openai_apikey")
        base_url=os.getenv("openai_baseurl")
    elif provider=="Anthropic":
        api_key=os.getenv("Anthropic_apikey")
        base_url=os.getenv("Anthropic_baseurl")
    elif provider=="deepseek":
        api_key=os.getenv("deepseek_apikey")
        base_url=os.getenv("deepseek_baseurl")
    elif provider=="auto":
        api_key=os.getenv("apikey")
        base_url=os.getenv("baseurl")
    else:
        raise ValueError("没有此服务商的相关信息")
    return api_key,base_url

# python -m core.LLM_extension 根目录下
if __name__=='__main__':
    providers=auto_detect_provider()
    print(f"建议使用如下服务商,但不承诺API调用一定成功\n{providers}\n")
    choice=input("请选择你的服务商:")
    llmex=LLMExtension(provider=choice)
    messages=[
        {
            "role":"user",
            "content":"你是什么模型"
        }
    ]
    llmex.think(messages)

        

