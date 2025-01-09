from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY=  os.getenv("Dp_Key")

BASE_URL = "https://api.deepseek.com"

class SingletonChatOpenAI:
    _instance = None
    # 如果不设置max_tokens，默认值为None，表示不限制最大token数
    # 但实际使用中，模型会有自己的最大token限制（通常是2048或4096）
    # 建议根据具体需求设置合理的max_tokens值
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SingletonChatOpenAI, cls).__new__(cls)
            cls._instance.llm = ChatOpenAI(
                model='deepseek-chat', 
                openai_api_key=API_KEY, 
                openai_api_base=BASE_URL,
                max_tokens=2048
            )
        return cls._instance

    def __getattr__(self, name):
        return getattr(self._instance.llm, name)

llm = SingletonChatOpenAI().llm


# 搞一个方法，对外开放吧
def get_completion_from_prompt(prompt,temperature=0.7):
    result =  llm.invoke(prompt,temperature=temperature)
    return result.content

#可传入消息列表
def get_completion_from_messages(messages,temperature=0):
    result =  llm.invoke(messages,temperature=temperature)
    return result.content




