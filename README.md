# Building Serverless Agents with Baseten Model APIs and Linkup

This guide walks through building a CLI agent that grounds its reasoning in real-time data using [Baseten Model APIs](https://baseten.co) (running DeepSeek-V3) and [Linkup](https://linkup.so) for live web search — all serverless, with no GPU infrastructure to manage.

## Prerequisites

* Python 3.8+ installed on your machine.
* A terminal (Command Prompt, PowerShell, or Terminal).

## Step 1: Configure the Environment

### 1. Initialize the project

```bash
mkdir baseten-linkup
cd baseten-linkup
```

### 2. Create a virtual environment

**Mac / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install openai linkup-sdk python-dotenv
```

## Step 2: Authentication

### 1. Generate a Baseten API Key

1. Navigate to your [Baseten settings](https://app.baseten.co/settings/api_keys).
2. Select **API Keys** from the sidebar.
3. Click **Create API key** and name it `linkup-agent`.
4. Copy the generated key.

### 2. Generate a Linkup API Key

1. Log in to the [Linkup Dashboard](https://app.linkup.so/).
2. Locate the **API Keys** section.
3. Copy your default API key.

### 3. Configure environment variables

Create a `.env` file in your `baseten-linkup` directory:

```
BASETEN_API_KEY=paste_your_baseten_key_here
LINKUP_API_KEY=paste_your_linkup_key_here
```

## Step 3: Build the Agent

Create a file named `agent.py` and add the following code:

```python
import os
import json
from datetime import datetime
from openai import OpenAI
from linkup import LinkupClient
from dotenv import load_dotenv

# ── Initialize clients ──────────────────────────────────────────────
# Baseten exposes an OpenAI-compatible endpoint, so we use the standard openai SDK.
# Linkup provides the web search capability.
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

# ── Tool schema ─────────────────────────────────────────────────────
# Tells the model a search_internet function exists. The model reads
# this schema and decides when to call it based on the user's query.
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

    # ── System prompt & conversation history ─────────────────────────
    # Instructs the model to prefer searching over stale training data.
    # We cap history at 10 turns here for context limits, but feel free
    # to adjust MAX_HISTORY_TURNS based on your use case.
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

            if len(history) > (MAX_HISTORY_TURNS * 2) + 1:
                history = [history[0]] + history[-(MAX_HISTORY_TURNS * 2):]

            # ── Keyword hints (anti-tool-fatigue) ────────────────────
            # In long conversations, models can stop calling tools even
            # when they should. If the query contains obvious search
            # keywords, we force tool_choice="required" to guarantee it.
            search_hints = [
                "latest", "current", "recent", "today", "now",
                "news", "stock price", "weather", "search", "look up",
                "next", "upcoming", "when is", "when does",
                "price", "cost", "how much", "buy", "worth",
            ]
            needs_search = any(h in user_input.lower() for h in search_hints)

            # ── Pass 1: Ask the model if it needs to search ──────────
            response = client.chat.completions.create(
                model=MODEL_SLUG,
                messages=history,
                tools=tools,
                tool_choice="required" if needs_search else "auto"
            )
            message = response.choices[0].message

            if message.tool_calls:
                history.append(message)

                # ── Execute Linkup search ────────────────────────────
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "search_internet":
                        args = json.loads(tool_call.function.arguments)
                        q = args.get("query")
                        print(f"Searching: {q}...")

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
                        except Exception as e:
                            content = f"Search error: {e}"

                        history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content
                        })

                # ── Pass 2: Synthesize search results into an answer ─
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
```

## Step 4: Run the Agent

```bash
python agent.py
```

### Validating the workflow

**Scenario 1: Internal Knowledge (no tool call)**

> **You:** What is the definition of philosophy?
> **Agent:** Philosophy is the study of fundamental questions about existence, knowledge, values, reason, mind, and language...

**Scenario 2: Tool-Augmented Reasoning**

> **You:** What are the latest books published on logic?
> **Agent:** `Searching: latest logic books 2025...`
> Agent synthesizes a response citing recent publications found via Linkup.
