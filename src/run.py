import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.agents import QueryAgent


async def run(query: str) -> dict:
    """Run the agent pipeline asynchronously on a single query."""
    agent = QueryAgent()
    result = await agent.ainvoke(query)
    return dict(result)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.run \"your question here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = asyncio.run(run(query))

    print("\n=== AGENT RESULT ===")
    print(f"\n[Input]        {result['input']}")
    print(f"[Analysis]     {result['analysis']}")
    print(f"\n[Final Answer]\n{result['final_response']}")
    print(f"\n[Outputs]      {result['outputs']}")


if __name__ == "__main__":
    main()
