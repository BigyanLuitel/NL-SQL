import os
import re
import json
import sqlite3

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from agent.state import State
from agent.tools import tools, DB_SCHEMA

llm = ChatGroq(
    model = "llama-3.3-70v-versatile",
    api_key = os.environ.get("GROQ_API_KEY"),   
    temperature = 0
    )
llm_with_tools = llm.bind_tools(tools)

evaluator = ChatGroq(
    model = "llama-3.3-70v-versatile",
    api_key = os.environ.get("GROQ_API_KEY"),
    temperature = 0
    )

def llm_node(state: State) -> dict:
    system = SystemMessage(content="""
    You are an expert SQL assistant.
    When the user asks a question, first call get_schema to retrieve
    the database schema, then generate a valid SQL query based on it.
    
    IMPORTANT: You are working with a SQLite database.
    SQLite date arithmetic rules:
    - Use date('now') for current date NOT CURRENT_DATE
    - Use date('now', '-N days') instead of INTERVAL N DAY
    - Use date('now', '+N days') for future dates
    
    Return the final SQL query clearly.
    """)
    
    response = llm_with_tools([system] + state["messages"])
    return {"messages":[response]}
def evaluator_node(state: State) -> dict:
    last_message = state["messages"][-1].content
    retry_count = state.get("retry_count", 0)
    schema = DB_SCHEMA
    eval_prompt = f"""You are a SQL expert and critic.

Given this database schema:
{schema}

Evaluate this SQL query:
{last_message}

Check for:
1. Correct table and column names from the schema
2. Valid SQL syntax
3. Does it actually answer the user's question?

Respond ONLY in this JSON format:
{{
    "is_valid": true or false,
    "issues": "describe issues if any or none",
    "corrected_sql": "corrected SQL if invalid or same SQL if valid"
}}"""