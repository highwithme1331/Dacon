#General Language Model Prompt Template
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(template)



#Conversational Language Model Prompt Template
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "{A}"),
    ("user", "{B}")
])

messages = chat_prompt.format_messages(
    A="message 1",
    B="message 2"
)



#OpenAI
from langchain_openai import ChatOpenAI

llm =  ChatOpenAI(model="gpt-4o-mini")

response = llm.invoke(message)



#Init_Chat_Model
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o-mini")

response = llm_openai.invoke(message)



#Chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "{A}"),
    ("user", "{B}")
])

llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

response = chain.invoke(message)