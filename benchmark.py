"""Benchmark: 100 queries testing tool-call judgment for DeepSeek-V3 + keyword hints."""
import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url="https://inference.baseten.co/v1")

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

# (query, should_search)
# True = should call tool, False = should NOT call tool
QUERIES = [
    # === SHOULD SEARCH: Obvious real-time queries ===
    ("What is Tesla's stock price today?", True),
    ("What's the weather in London right now?", True),
    ("Who won the NBA game last night?", True),
    ("What are the latest headlines from Reuters?", True),
    ("How much does a MacBook Pro cost?", True),
    ("What is Bitcoin trading at currently?", True),
    ("When is the next Apple keynote?", True),
    ("What movies are playing in theaters right now?", True),
    ("What's the current unemployment rate in the US?", True),
    ("Who is leading in the latest election polls?", True),

    # === SHOULD SEARCH: Tricky — no obvious keywords ===
    ("Did Elon Musk tweet anything controversial?", True),
    ("Has OpenAI released GPT-5 yet?", True),
    ("Is there a ceasefire in Ukraine?", True),
    ("Are interest rates going up or down?", True),
    ("How many subscribers does Netflix have?", True),
    ("What's Spotify's most streamed song?", True),
    ("Who is the CEO of Twitter?", True),
    ("Did the Fed cut rates?", True),
    ("Is Costco open on Easter this year?", True),
    ("What happened at the Oscars?", True),

    # === SHOULD SEARCH: Pricing and shopping ===
    ("How much is a PS5 right now?", True),
    ("What's the cheapest flight from NYC to LA?", True),
    ("How much does a Toyota Camry cost?", True),
    ("What's the price of gold per ounce?", True),
    ("How much is a gallon of gas in California?", True),

    # === SHOULD SEARCH: Upcoming events ===
    ("When does the new season of Stranger Things come out?", True),
    ("When is the next UFC fight?", True),
    ("What concerts are happening in NYC this weekend?", True),
    ("When does daylight saving time start?", True),
    ("When is the next SpaceX launch?", True),

    # === SHOULD SEARCH: Very recent or breaking ===
    ("Was there an earthquake today?", True),
    ("Any flight delays at JFK airport?", True),
    ("Is the stock market open today?", True),
    ("What's trending on Twitter right now?", True),
    ("Are there any active wildfires in California?", True),

    # === SHOULD SEARCH: Edge cases that try to fool the model ===
    ("Tell me about the Mars rover's latest discovery", True),
    ("What's going on with the TikTok ban?", True),
    ("Any recalls on Tesla vehicles?", True),
    ("How's the housing market doing?", True),
    ("What did Congress vote on this week?", True),
    ("Is Amazon Prime Day happening soon?", True),
    ("What's the exchange rate for USD to EUR?", True),
    ("How long is the wait at the DMV near me?", True),
    ("What restaurants got Michelin stars this year?", True),
    ("Has NVIDIA announced a new GPU?", True),
    ("Score of the Yankees game", True),
    ("Who got eliminated on The Bachelor?", True),
    ("Ethereum merge status", True),
    ("COVID cases this week", True),
    ("Box office numbers this weekend", True),

    # === SHOULD NOT SEARCH: Pure knowledge / reasoning ===
    ("What is the speed of light?", False),
    ("Explain quantum entanglement in simple terms", False),
    ("What's the capital of Mongolia?", False),
    ("How does a diesel engine differ from a gasoline engine?", False),
    ("What causes the northern lights?", False),
    ("Solve: 2x + 5 = 17", False),
    ("Write a limerick about a cat", False),
    ("What year did World War 2 end?", False),
    ("Explain the difference between TCP and UDP", False),
    ("What is the Pythagorean theorem?", False),

    # === SHOULD NOT SEARCH: Creative / conversational ===
    ("Write me a poem about autumn", False),
    ("Give me 5 dinner ideas with chicken", False),
    ("Help me write a professional email declining a meeting", False),
    ("Create a workout plan for a beginner", False),
    ("Tell me a joke about programmers", False),
    ("What's the difference between empathy and sympathy?", False),
    ("Explain blockchain like I'm 5", False),
    ("Give me a metaphor for persistence", False),
    ("What are the pros and cons of remote work?", False),
    ("Summarize the plot of 1984 by George Orwell", False),

    # === SHOULD NOT SEARCH: Math / logic / trivia ===
    ("What's 15% of 340?", False),
    ("How many planets are in the solar system?", False),
    ("What does 'caveat emptor' mean?", False),
    ("Convert 100 Fahrenheit to Celsius", False),
    ("What is the Fibonacci sequence?", False),
    ("Who painted the Mona Lisa?", False),
    ("What's the chemical formula for water?", False),
    ("How many bones are in the human body?", False),
    ("What's the longest river in the world?", False),
    ("Explain the difference between affect and effect", False),

    # === SHOULD NOT SEARCH: Philosophical / abstract ===
    ("What is the meaning of life?", False),
    ("Is free will real?", False),
    ("What makes a good leader?", False),
    ("Can machines truly think?", False),
    ("What is consciousness?", False),

    # === EDGE CASES: Ambiguous — could go either way ===
    ("What's the best programming language to learn?", False),
    ("How do I fix a leaky faucet?", False),
    ("What should I name my dog?", False),
    ("Is Python better than JavaScript?", False),
    ("How do I negotiate a raise?", False),
]

def needs_keyword_search(query):
    return any(h in query.lower() for h in search_hints)

print(f"Benchmarking {len(QUERIES)} queries on {MODEL_SLUG}")
print(f"System prompt: middle-ground + keyword hints (tool_choice=required)\n")

passed = 0
failed = 0
errors = 0
results = []

for i, (query, should_search) in enumerate(QUERIES):
    keyword_forced = needs_keyword_search(query)
    tool_choice = "required" if keyword_forced else "auto"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL_SLUG,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        called = bool(resp.choices[0].message.tool_calls)
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
        print(f"  {i+1:3d}. {status}{hint_tag} | exp={expected:9s} got={actual:9s} | {query}")

        results.append({
            "query": query,
            "expected": should_search,
            "actual": called,
            "correct": correct,
            "keyword_forced": keyword_forced,
        })

    except Exception as e:
        errors += 1
        print(f"  {i+1:3d}. ERROR | {query} | {e}")
        results.append({
            "query": query,
            "expected": should_search,
            "actual": None,
            "correct": False,
            "keyword_forced": keyword_forced,
            "error": str(e),
        })

# Summary
total = len(QUERIES)
should_search_count = sum(1 for _, s in QUERIES if s)
should_not_count = total - should_search_count

search_correct = sum(1 for r in results if r["expected"] and r["correct"])
nosearch_correct = sum(1 for r in results if not r["expected"] and r["correct"])
keyword_saves = sum(1 for r in results if r.get("keyword_forced") and r["expected"] and r["correct"])

print(f"\n{'='*60}")
print(f"RESULTS: {passed}/{total} correct ({passed/total*100:.1f}%)")
print(f"  Should search:     {search_correct}/{should_search_count} correct")
print(f"  Should NOT search: {nosearch_correct}/{should_not_count} correct")
print(f"  Keyword hint saves: {keyword_saves} queries forced via tool_choice=required")
print(f"  Errors: {errors}")
print(f"{'='*60}")

# Save detailed results
with open("benchmark_results.json", "w") as f:
    json.dump({"model": MODEL_SLUG, "total": total, "passed": passed, "failed": failed, "errors": errors, "results": results}, f, indent=2)
print(f"\nDetailed results saved to benchmark_results.json")
