"""Run ReAct with Azure OpenAI"""

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_openai import AzureChatOpenAI

load_dotenv()


llm = AzureChatOpenAI(
    openai_api_version="2024-02-15-preview",
    azure_deployment="gpt-4",
)
# ツールを定義
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wikipedia]
# ReAct用のプロンプトを用意（0-shotのReAct用のプロンプトです。）
prompt = hub.pull("hwchase17/react")
# Agentを作成
agent = create_react_agent(llm, tools, prompt)
# AgentExecutorを作成
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# 処理開始
agent_executor.invoke({"input": "2023年に開催したWorld Baseball Classicの優勝国は？"})
