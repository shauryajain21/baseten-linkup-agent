import os
import json
from datetime import datetime
from openai import OpenAI
from linkup import LinkupClient
from dotenv import load_dotenv

load_dotenv()
BASETEN_API_KEY = os.environ.get("BASETEN_API_KEY")
LINKUP_API_KEY = os.environ.get("LINKUP_API_KEY")

if not BASETEN_API_KEY or not LINKUP_API_KEY:
    print("Error: API keys not found. Please check your .env file.")
    exit(1)

MODEL_SLUG = "deepseek-ai/DeepSeek-V3-0324"

client = OpenAI(
    api_key=BASETEN_API_KEY,
    base_url="https://inference.baseten.co/v1"
)
linkup = LinkupClient(api_key=LINKUP_API_KEY)

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": "Search the web for real-time information: news, prices, weather, recent events, or any factual query.",
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
    }
]

def main():
    print(f"--- Serverless Agent ({MODEL_SLUG}) ---")
    print("Type 'quit' to exit.\n")

    today_str = datetime.now().strftime("%B %d, %Y")
    system_prompt = (
        f"You are a helpful assistant. Today is {today_str}. "
        f"You have a search_internet tool that retrieves live web results via Linkup.\n\n"
        f"Always use it when the user asks about anything time-sensitive: "
        f"news, stock prices, weather, sports scores, recent releases, or current events. "
        f"Also use it whenever the query contains words like 'latest', 'current', 'recent', 'today', or 'now'. "
        f"Prefer searching over relying on your training data for anything that could be outdated."
    )

    history = [{"role": "system", "content": system_prompt}]
    MAX_HISTORY_TURNS = 10

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            history.append({"role": "user", "content": user_input})

            # Trim history to stay within context limits
            if len(history) > (MAX_HISTORY_TURNS * 2) + 1:
                history = [history[0]] + history[-(MAX_HISTORY_TURNS * 2):]

            # Force tool use when the query clearly needs live data
            search_hints = [
                "latest", "current", "recent", "today", "now",
                "news", "stock price", "weather", "search", "look up",
                "next", "upcoming", "when is", "when does",
                "price", "cost", "how much", "buy", "worth",
            ]
            needs_search = any(h in user_input.lower() for h in search_hints)

            # Pass 1: Decide whether to call a tool
            response = client.chat.completions.create(
                model=MODEL_SLUG,
                messages=history,
                tools=tools,
                tool_choice="required" if needs_search else "auto"
            )
            message = response.choices[0].message

            if message.tool_calls:
                history.append(message)

                for tc in message.tool_calls:
                    q = json.loads(tc.function.arguments)["query"]
                    print(f"Searching: {q}...")
                    try:
                        result = linkup.search(query=q, depth="standard", output_type="searchResults")
                        content = "\n\n".join(
                            f"{r.name}\n{r.url}\n{r.content}" for r in result.results
                        )
                    except Exception as e:
                        content = f"Search error: {e}"
                    history.append({"role": "tool", "tool_call_id": tc.id, "content": content})

                # Pass 2: Synthesize search results into final answer
                final = client.chat.completions.create(
                    model=MODEL_SLUG,
                    messages=history
                )
                reply = final.choices[0].message
                print(f"Agent: {reply.content}\n")
                history.append(reply)

            else:
                print(f"Agent: {message.content}\n")
                history.append(message)

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
