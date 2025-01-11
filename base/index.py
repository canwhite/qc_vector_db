from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pathlib import Path


root_path = str(Path(__file__).parent.parent)
import sys 
sys.path.append(root_path)
from llm import llm
print(llm)


# schema and parser
class Task(BaseModel):
    task_name: str = Field(description="分析任务，得到任务名称")
    task_type: str = Field(description="分析任务，得到任务的类型")
    task_content: str = Field(description="分析任务，得到任务的内容")
    task_time: str = Field(description="分析任务，得到任务的时间")
    task_result: str = Field(description="实现任务，得到组件代码部分,注意这里给代码")
    task_status: str = Field(description="分析任务，得到任务的状态,完成或未完成")

parser = JsonOutputParser(pydantic_object=Task)


prompt = PromptTemplate(
    template="根据用户输入的问题得到任务JSON.\n{instructions}\n{query}\n", # 固定文本和占位符
    input_variables=["query"], # 外部输入的变量列表
    partial_variables={"instructions": parser.get_format_instructions()}, # 于定义变量，格式说明
)

print(prompt)


# chain
task_chain = prompt | llm | parser


task_query = "帮我写一个查询组件，react的"
# invoke
task_data = task_chain.invoke({"query": task_query})
print(task_data) 




