"""Run ReAct with Anthropic"""

import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.google_search.tool import GoogleSearchRun
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_anthropic import ChatAnthropic

load_dotenv()

llm = ChatAnthropic(model_name=os.getenv("ANTHROPIC_MODEL"))
tools = [GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())]
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
)
with open("input.txt", "r", encoding="utf-8") as file:
    input_text = file.read().replace("\n", "\\n")
    agent_executor.invoke({"input": input_text})
