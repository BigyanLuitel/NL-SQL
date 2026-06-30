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

# --- LLMs ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

evaluator_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

tool_node = ToolNode(tools)


# --- Node 1: LLM ---
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
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


# --- Node 2: Evaluator ---
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

    response = evaluator_llm.invoke([HumanMessage(content=eval_prompt)])
    clean = re.sub(r"```json|```", "", response.content).strip()

    if not clean:
        print("⚠️ Evaluator returned empty response, passing SQL through")
        return {
            "messages": [AIMessage(content=last_message)],
            "sql_query": last_message,
            "retry_count": 0
        }

    try:
        evaluation = json.loads(clean)
    except json.JSONDecodeError:
        print(" Could not parse evaluator response")
        return {
            "messages": [AIMessage(content=last_message)],
            "sql_query": last_message,
            "retry_count": 0
        }

    if evaluation["is_valid"]:
        print(" SQL is valid!")
        return {
            "messages": [AIMessage(content=evaluation["corrected_sql"])],
            "sql_query": evaluation["corrected_sql"],
            "retry_count": 0
        }
    else:
        print(f" Issues found: {evaluation['issues']}")
        return {
            "messages": [AIMessage(content=f"The SQL was invalid. Issues: {evaluation['issues']}. Please regenerate.")],
            "retry_count": retry_count + 1
        }


# --- Node 3: Executor ---
def executor_node(state: State) -> dict:
    sql = state.get("sql_query") or state["messages"][-1].content

    try:
        conn = sqlite3.connect("database/library.db")
        df = pd.read_sql_query(sql, conn)
        conn.close()

        if df.empty:
            return {"query_results": "No data found."}

        print(f" Executor: {len(df)} rows, columns: {list(df.columns)}")
        return {"query_results": df.to_string(index=False)}

    except Exception as e:
        error_msg = str(e)
        print(f" Executor error: {error_msg}")
        return {"query_results": f"SQL Error: {error_msg}"}