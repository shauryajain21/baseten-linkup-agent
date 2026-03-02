"""Non-interactive test script for the Baseten + Linkup agent."""
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
    print("Error: API keys not found.")
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
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    }
]

today_str = datetime.now().strftime("%B %d, %Y")
system_prompt = (
    f"You are a helpful assistant. Today is {today_str}. "
    f"You have a search_internet tool that retrieves live web results via Linkup.\n\n"
    f"Always use it when the user asks about anything time-sensitive: "
    f"news, stock prices, weather, sports scores, recent releases, or current events. "
    f"Also use it whenever the query contains words like 'latest', 'current', 'recent', 'today', or 'now'. "
    f"Prefer searching over relying on your training data for anything that could be outdated."
)

def run_query(query):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print('='*60)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=MODEL_SLUG,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    message = response.choices[0].message

    if message.tool_calls:
        print(f"[TOOL CALL DETECTED]")
        messages.append(message)

        for tool_call in message.tool_calls:
            if tool_call.function.name == "search_internet":
                args = json.loads(tool_call.function.arguments)
                q = args.get("query")
                print(f"  Searching: {q}...")

                try:
                    result = linkup.search(
                        query=q,
                        depth="standard",
                        output_type="searchResults"
                    )
                    content = "\n\n".join([
                        f"Title: {r.name}\nURL: {r.url}\nContent: {r.content}"
                        for r in result.results
                    ])
                    print(f"  Got {len(result.results)} results")
                except Exception as e:
                    content = f"Search error: {e}"
                    print(f"  Search failed: {e}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content
                })

        final = client.chat.completions.create(
            model=MODEL_SLUG,
            messages=messages
        )
        reply = final.choices[0].message.content
        print(f"\nAGENT RESPONSE:\n{reply[:500]}...")
        return True  # tool was called
    else:
        print(f"[NO TOOL CALL]")
        print(f"\nAGENT RESPONSE:\n{message.content[:500]}...")
        return False  # no tool call

# --- RUN TESTS ---
print("Testing Baseten + Linkup Agent")
print(f"Model: {MODEL_SLUG}\n")

# Test 1: Should NOT trigger a tool call
print("\n--- TEST 1: Internal knowledge (no tool expected) ---")
used_tool_1 = run_query("What is the definition of philosophy?")
print(f"\nResult: {'PASS' if not used_tool_1 else 'UNEXPECTED - tool was called'}")

# Test 2: Should trigger a tool call
print("\n\n--- TEST 2: Real-time query (tool expected) ---")
used_tool_2 = run_query("What are the latest AI news headlines today?")
print(f"\nResult: {'PASS' if used_tool_2 else 'FAIL - tool was NOT called'}")

print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"  Test 1 (no tool): {'PASS' if not used_tool_1 else 'UNEXPECTED'}")
print(f"  Test 2 (tool call): {'PASS' if used_tool_2 else 'FAIL'}")
print('='*60)
