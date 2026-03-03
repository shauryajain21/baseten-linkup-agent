# Building a Serverless Agent with GPT-OSS 120B and Linkup

This guide walks through building a CLI agent that grounds its reasoning in real-time data using [Baseten Model APIs](https://baseten.co) (running GPT-OSS 120B) and [Linkup](https://linkup.so) for live web search — all serverless, with no GPU infrastructure to manage.

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
pip install openai linkup-sdk
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

**Mac / Linux:**

```bash
export BASETEN_API_KEY=paste_your_baseten_key_here
export LINKUP_API_KEY=paste_your_linkup_key_here
```

**Windows:**

```bash
set BASETEN_API_KEY=paste_your_baseten_key_here
set LINKUP_API_KEY=paste_your_linkup_key_here
```

## Step 3: Build the Agent

Create a file named `agent.py` and add the following code:

```python
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
```

## Step 4: Run the Agent

```bash
python gpt-oss/agent.py
```

### Validating the workflow

**Scenario 1: Internal Knowledge (no tool call)**

> **You:** What is the definition of philosophy?
> **Agent:** Philosophy is the study of fundamental questions about existence, knowledge, values, reason, mind, and language...

**Scenario 2: Tool-Augmented Reasoning**

> **You:** What are the latest books published on logic?
> Searching using Linkup: latest logic books 2025...
> **Agent:** Synthesizes a response citing recent publications found via Linkup.
