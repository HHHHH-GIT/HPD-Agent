import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


def get_llm(
    model: str = "qwen-plus",
    temperature: float = 0,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    stream: bool = False,
) -> ChatOpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set.")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        stream=stream,
    )


def get_structured_llm(
    schema: type[BaseModel],
    model: str = "qwen-plus",
    temperature: float = 0,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> ChatOpenAI:
    """Return an LLM bound to output a specific Pydantic schema."""
    llm = get_llm(model=model, temperature=temperature, base_url=base_url)
    return llm.with_structured_output(schema, method="json_schema")
