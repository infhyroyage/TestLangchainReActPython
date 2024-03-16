"""Run ReAct with Azure OpenAI and """

import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.google_search.tool import GoogleSearchRun
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_openai import AzureChatOpenAI

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_version="2024-02-15-preview",
    azure_deployment=os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
)
tools = [GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())]
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
)
agent_executor.invoke({"input": "2023年に開催したWorld Baseball Classicの優勝国は？"})
