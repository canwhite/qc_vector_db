from smolagents import OpenAIServerModel, CodeAgent, DuckDuckGoSearchTool
import sys
from pathlib import Path
from queue import Queue
import threading

root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)
from dotenv import load_dotenv
load_dotenv()
import os

api_key = os.getenv("Dp_Key")

class AgentPool:
    _model = OpenAIServerModel(
        model_id="deepseek-chat",
        api_base="https://api.deepseek.com",
        api_key=api_key
    )
    
    def __init__(self, max_size=5, tools=None):
        self._pool = Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._tools = tools or []
        
        # Initialize pool with agents
        for _ in range(max_size):
            agent = CodeAgent(tools=self._tools, model=self._model)
            self._pool.put(agent)

    def get_agent(self):
        with self._lock:
            if self._pool.empty():
                return CodeAgent(tools=self._tools, model=self._model)
            return self._pool.get()

    def release_agent(self, agent):
        with self._lock:
            if self._pool.qsize() < self._pool.maxsize:
                self._pool.put(agent)

    def __enter__(self):
        return self.get_agent()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_agent(self)

# 创建agent池
agent_pool = AgentPool()

# 使用agent池
with agent_pool as agent:
    res = agent.run("how to implement a chatbot")
    print("--", res)
