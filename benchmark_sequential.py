"""Sequential benchmark: 30 queries in one conversation to test tool fatigue."""
import os
import json
from datetime import datetime
from openai import OpenAI
from linkup import LinkupClient
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url="https://inference.baseten.co/v1")
linkup_client = LinkupClient(api_key=os.environ["LINKUP_API_KEY"])

MODEL_SLUG = "deepseek-ai/DeepSeek-V3-0324"

today_str = datetime.now().strftime("%B %d, %Y")
system_prompt = (
    f"You are a helpful assistant. Today is {today_str}. "
    f"You have a search_internet tool that retrieves live web results via Linkup.\n\n"
    f"Always use it when the user asks about anything time-sensitive: "
    f"news, stock prices, weather, sports scores, recent releases, or current events. "
    f"Also use it whenever the query contains words like 'latest', 'current', 'recent', 'today', or 'now'. "
    f"Prefer searching over relying on your training data for anything that could be outdated."
)

tools = [{
    "type": "function",
    "function": {
        "name": "search_internet",
        "description": "Search the web for real-time information: news, prices, weather, recent events, or any factual query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"]
        }
    }
}]

search_hints = [
    "latest", "current", "recent", "today", "now",
    "news", "stock price", "weather", "search", "look up",
    "next", "upcoming", "when is", "when does",
    "price", "cost", "how much", "buy", "worth",
]

# 30 queries alternating search/no-search to stress-test tool fatigue
QUERIES = [
    ("What is NVIDIA's stock price today?", True),
    ("What is the Pythagorean theorem?", False),
    ("Who won the most recent Super Bowl?", True),
    ("Write me a haiku about rain", False),
    ("What's the weather in Tokyo right now?", True),
    ("Explain how photosynthesis works", False),
    ("Has OpenAI released GPT-5 yet?", True),
    ("What's 25% of 480?", False),
    ("How much does a PS5 cost?", True),
    ("What makes a good leader?", False),
    ("What did Congress vote on this week?", True),
    ("Tell me a joke about cats", False),
    ("Score of the Lakers game", True),
    ("What is the speed of light?", False),
    ("Any flight delays at JFK airport?", True),
    ("Convert 100 Celsius to Fahrenheit", False),
    ("What's trending on Twitter right now?", True),
    ("Who painted the Mona Lisa?", False),
    ("Is there a ceasefire in Ukraine?", True),
    ("Summarize the plot of Hamlet", False),
    ("What's the exchange rate for USD to EUR?", True),
    ("What causes tides on Earth?", False),
    ("Did the Fed cut rates?", True),
    ("Give me 3 dinner ideas with salmon", False),
    ("What new features did Apple announce most recently?", True),
    ("What year did World War 2 end?", False),
    ("Are there any active wildfires in California?", True),
    ("Explain blockchain like I'm 5", False),
    ("What's the price of gold per ounce?", True),
    ("Is free will real?", False),
]

MAX_HISTORY_TURNS = 10
history = [{"role": "system", "content": system_prompt}]

print(f"Sequential benchmark: {len(QUERIES)} queries in ONE conversation")
print(f"Model: {MODEL_SLUG}")
print(f"History limit: {MAX_HISTORY_TURNS} turns\n")

passed = 0
failed = 0
errors = 0

for i, (query, should_search) in enumerate(QUERIES):
    keyword_forced = any(h in query.lower() for h in search_hints)
    tool_choice = "required" if keyword_forced else "auto"

    history.append({"role": "user", "content": query})

    # Trim history (same logic as agent.py)
    if len(history) > (MAX_HISTORY_TURNS * 2) + 1:
        history = [history[0]] + history[-(MAX_HISTORY_TURNS * 2):]

    try:
        resp = client.chat.completions.create(
            model=MODEL_SLUG,
            messages=history,
            tools=tools,
            tool_choice=tool_choice,
        )
        message = resp.choices[0].message
        called = bool(message.tool_calls)

        if called:
            # Execute the search and do Pass 2 (like agent.py)
            history.append(message)
            for tc in message.tool_calls:
                if tc.function.name == "search_internet":
                    args = json.loads(tc.function.arguments)
                    q = args.get("query")
                    try:
                        result = linkup_client.search(query=q, depth="standard", output_type="searchResults")
                        content = "\n\n".join([
                            f"Title: {r.name}\nURL: {r.url}\nContent: {r.content}"
                            for r in result.results
                        ])
                    except Exception as e:
                        content = f"Search error: {e}"
                    history.append({"role": "tool", "tool_call_id": tc.id, "content": content})

            final = client.chat.completions.create(model=MODEL_SLUG, messages=history)
            reply = final.choices[0].message
            history.append(reply)
        else:
            history.append(message)

        correct = (called == should_search)
        if correct:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "MISS"

        hint_tag = " [KW]" if keyword_forced else ""
        expected = "SEARCH" if should_search else "NO SEARCH"
        actual = "SEARCH" if called else "NO SEARCH"
        hist_len = len(history)
        print(f"  {i+1:3d}. {status}{hint_tag} | exp={expected:9s} got={actual:9s} | hist={hist_len:3d} | {query}")

    except Exception as e:
        errors += 1
        print(f"  {i+1:3d}. ERROR | {query} | {e}")

total = len(QUERIES)
should_search_count = sum(1 for _, s in QUERIES if s)
should_not_count = total - should_search_count
search_correct = sum(1 for i, (_, s) in enumerate(QUERIES) if s and i < passed + failed)

print(f"\n{'='*60}")
print(f"RESULTS: {passed}/{total} correct ({passed/total*100:.1f}%)")
print(f"  Failed: {failed}  Errors: {errors}")
print(f"{'='*60}")
