"""Run ReAct with Azure OpenAI and Google Search"""

import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun
from langchain_openai import AzureChatOpenAI

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
)
tools = [GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())]
prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
)
chat_history = []

with open("input.txt", "r", encoding="utf-8") as file:
    input_text = file.read().replace("\n", "\\n")

    while True:
        result = agent_executor.invoke(
            {"input": input_text, "chat_history": chat_history}
        )
        chat_history.extend(
            [HumanMessage(content=input_text), AIMessage(content=result["output"])]
        )

        try:
            input_text = input("Enter additional input (or press Ctrl+C to exit): ")
        except KeyboardInterrupt:
            print("\nTerminated by user.")
            break
