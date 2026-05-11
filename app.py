import logging
import os
import warnings

import streamlit as st

from agent import ConversationMemory, get_sql_agent, run_agent_query
from config import settings
from db import get_database


def setup_logging() -> None:
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    warnings.filterwarnings("ignore", message=r".*Accessing `__path__`.*")

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(settings.log_file)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler], force=True)
    for noisy in ["transformers", "sentence_transformers", "httpx", "urllib3", "sqlalchemy.engine", "langchain"]:
        logging.getLogger(noisy).setLevel(logging.ERROR)


@st.cache_resource(show_spinner=False)
def load_database():
    return get_database()


@st.cache_resource(show_spinner=False)
def load_agent():
    db = load_database()
    return get_sql_agent(db)


def initialize_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me anything about your data."}
        ]
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()


def main() -> None:
    setup_logging()
    st.set_page_config(page_title="CRM AI Assistant", page_icon=":speech_balloon:", layout="wide")
    st.title("CRM AI Assistant")
    st.caption("Powered by Groq · Fallback: Ollama qwen2.5:14b")

    initialize_session()
    agent = load_agent()
    memory: ConversationMemory = st.session_state.memory

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("meta")
            if meta:
                st.caption(
                    f"Latency: {meta['latency_ms']} ms | "
                    f"Confidence: {meta['confidence']:.2f} | "
                    f"Tables: {', '.join(meta['tables_used']) if meta['tables_used'] else 'None'}"
                )
                if meta.get("sql_queries"):
                    with st.expander("Executed SQL"):
                        for q in meta["sql_queries"]:
                            st.code(q, language="sql")

    prompt = st.chat_input("Ask a question about your CRM data...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_agent_query(agent, prompt, memory=memory)
        st.markdown(result["answer"])
        st.caption(
            f"Latency: {result['latency_ms']} ms | "
            f"Confidence: {result['confidence']:.2f} | "
            f"Tables: {', '.join(result['tables_used']) if result['tables_used'] else 'None'}"
        )
        if result.get("_sub_results"):
            with st.expander("Query Plan & Sub-results"):
                for i, item in enumerate(result["_sub_results"], 1):
                    st.markdown(f"**Part {i}: {item['sub_query']}** _(intent: {item['intent']})_")
                    sqls = item.get("data", {}).get("sql_queries", [])
                    for sq in sqls:
                        st.code(sq, language="sql")
        elif result.get("sql_queries"):
            with st.expander("Executed SQL"):
                for q in result["sql_queries"]:
                    st.code(q, language="sql")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": {
            "latency_ms": result["latency_ms"],
            "confidence": result["confidence"],
            "tables_used": result["tables_used"],
            "sql_queries": result.get("sql_queries", []),
        },
    })


if __name__ == "__main__":
    main()
