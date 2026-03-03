import os
import json
from datetime import datetime
import anthropic
from linkup import LinkupClient

claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
linkup_client = LinkupClient(api_key=os.environ.get("LINKUP_API_KEY"))

tools = [{
    "name": "search_web",
    "description": "Search the web in real time. Use this tool whenever the user needs trusted facts, news, or source-backed information. Returns comprehensive content from the most relevant sources.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            }
        },
        "required": ["query"]
    }
}]

def main():
    print("--- Claude + Linkup Agent ---")
    print("Type 'quit' to exit.\n")

    today_str = datetime.now().strftime("%B %d, %Y")
    system_prompt = f"You are a helpful assistant. Today is {today_str}. Use web search when you need current information."

    messages = []

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            messages.append({"role": "user", "content": user_input})

            response = claude_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )

            if response.stop_reason == "tool_use":
                tool_use = next(block for block in response.content if block.type == "tool_use")

                print(f"Searching using Linkup: {tool_use.input['query']}...")
                linkup_response = linkup_client.search(query=tool_use.input["query"], depth="standard", output_type="searchResults")
                search_results = json.dumps([{"content": result.content} for result in linkup_response.results])

                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": search_results
                    }]
                })

                final_response = claude_client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                )

                text_content = next((block.text for block in final_response.content if hasattr(block, "text")), None)
                print(f"Agent: {text_content}\n")
                messages.append({"role": "assistant", "content": final_response.content})
            else:
                text_content = next((block.text for block in response.content if hasattr(block, "text")), None)
                print(f"Agent: {text_content}\n")
                messages.append({"role": "assistant", "content": response.content})

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
