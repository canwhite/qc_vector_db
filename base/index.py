import logging
import sys
from pathlib import Path
from typing import Dict, Any

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure paths
root_path = str(Path(__file__).parent.parent)
sys.path.append(root_path)

from llm import llm

#实现

# Define models
class Task(BaseModel):
    """Task model representing a task to be processed"""
    task_name: str = Field(description="分析任务，得到任务名称")
    task_type: str = Field(description="分析任务，得到任务的类型")
    task_content: str = Field(description="分析任务，得到任务的内容")
    task_time: str = Field(description="分析任务，得到任务的时间")
    task_result: str = Field(description="实现任务，得到组件代码部分,注意这里给代码")
    task_status: str = Field(description="分析任务，得到任务的状态,完成或未完成")

# Initialize parser
parser = JsonOutputParser(pydantic_object=Task)

# Configure prompt template
prompt = PromptTemplate(
    template="根据用户输入的问题得到任务JSON.\n{instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"instructions": parser.get_format_instructions()},
)

logger.info("Initialized prompt template: %s", prompt)


def process_task(query: str) -> Dict[str, Any]:
    """Process a task query and return the parsed result"""
    try:
        # Create and execute chain
        task_chain = prompt | llm | parser
        return task_chain.invoke({"query": query})
    except Exception as e:
        logger.error("Error processing task: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        task_query = "帮我写一个查询组件，react的"
        task_data = process_task(task_query)
        logger.info("Task processed successfully: %s", task_data)
    except Exception as e:
        logger.error("Failed to process task: %s", str(e))
