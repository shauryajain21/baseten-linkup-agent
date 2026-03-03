import os
import json
from datetime import datetime
from openai import OpenAI
from linkup import LinkupClient

client = OpenAI(
    api_key=os.environ.get("BASETEN_API_KEY"),
    base_url="https://inference.baseten.co/v1"
)
linkup_client = LinkupClient(api_key=os.environ.get("LINKUP_API_KEY"))

tools = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web in real time. Use this tool whenever the user needs trusted facts, news, or source-backed information. Returns comprehensive content from the most relevant sources.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}]

def main():
    print("--- GPT-OSS 120B + Linkup ---")
    print("Type 'quit' to exit.\n")

    today_str = datetime.now().strftime("%B %d, %Y")
    system_prompt = (
        f"You are a helpful assistant. Today is {today_str}. "
        f"Use web search when you need current information. "
        f"Prefer searching over relying on your training data for anything that could be outdated."
    )

    history = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            history.append({"role": "user", "content": user_input})

            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=history,
                tools=tools,
                tool_choice="auto"
            )
            message = response.choices[0].message

            while message.tool_calls:
                history.append(message)
                for tc in message.tool_calls:
                    q = json.loads(tc.function.arguments)["query"]
                    print(f"Searching using Linkup: {q}...")
                    try:
                        result = linkup_client.search(query=q, depth="standard", output_type="searchResults")
                        content = "\n\n".join(
                            f"{r.name}\n{r.url}\n{r.content}" for r in result.results
                        )
                    except Exception as e:
                        content = f"Search error: {e}"
                    history.append({"role": "tool", "tool_call_id": tc.id, "content": content})

                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=history,
                    tools=tools,
                    tool_choice="auto"
                )
                message = response.choices[0].message

            print(f"Agent: {message.content}\n")
            history.append(message)

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
