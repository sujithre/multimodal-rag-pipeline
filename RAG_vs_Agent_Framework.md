# RAG Query vs. Agent Framework — Comparison

## The Question

Step 4 (`04_rag_query.ipynb`) implements a classic RAG pipeline: embed the query, search the index, stitch a prompt, call the LLM. It works — so why bother with an agent framework?

This document compares the **manual RAG approach** in Step 4 with the three **Agent Framework approaches** in Steps 5–7 and explains when and why the agent-based architecture is the better choice.

---

## What Is Microsoft Agent Framework?

[Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/) is a lightweight, open-source SDK for building AI agents. An **agent** is an LLM + instructions + tools that can:

- Decide **which tool** to call (and when) based on the user's question
- Call tools **multiple times** or in **sequence** to gather information
- Compose a final answer from tool outputs automatically
- Maintain **conversation state** across turns

Instead of you writing the glue code (embed → search → build prompt → call LLM), the agent does it by itself. You just define the tools and instructions.

---

## Step 4: Manual RAG Pipeline (No Agent)

```
User Question
     │
     ▼
Your code: embed query ──▶ hybrid search ──▶ build context ──▶ call GPT-4.1
     │
     ▼
   Answer
```

### How it works

```python
def ask(question: str, top_k: int = 5) -> str:
    chunks = search(question, top_k=top_k)       # You embed, search, collect
    context = "\n\n---\n\n".join(...)             # You stitch the prompt
    response = openai_client.chat.completions.create(...)  # You call the LLM
    return response.choices[0].message.content
```

Everything is **your responsibility**: embedding the query, running the search, formatting results into a prompt, calling the chat API, and returning the answer. The LLM has no autonomy — it only sees what you put in the prompt.

### Strengths

| ✅ | Benefit |
|---|---|
| Simple | ~40 lines of code, easy to understand and debug |
| Predictable | Every query follows the exact same path |
| Fast | One search call + one LLM call, no overhead |
| No extra dependencies | Just `openai` + `azure-search-documents` |

### Limitations

| ❌ | Limitation |
|---|---|
| **Single-shot retrieval** | The LLM can't say "those results weren't helpful, let me search again with different terms" |
| **No multi-turn** | Each call is stateless — follow-up questions don't have context from previous answers |
| **Rigid workflow** | The pipeline is hardcoded: always embed → search → generate. It can't adapt |
| **Manual prompt engineering** | You manually stitch context into the prompt; the LLM has no control over what it retrieves |
| **Scaling pain** | Adding a second tool (e.g., a calculator, a database lookup) means rewriting the pipeline |

---

## Step 5: Agent Framework + OpenAI Client (Local Agent)

```
User Question
     │
     ▼
Agent (instructions + tools)
     │
     ├──▶ Decides: "I need to search" ──▶ calls search_books tool
     │                                          │
     │◀─── tool results ◀──────────────────────┘
     │
     ├──▶ Decides: "I have enough info" ──▶ generates answer
     │
     ▼
   Answer with citations
```

### How it works

```python
@tool(name="search_books", description="Search the Harry Potter books...")
def search_books(query: str, top_k: int = 5) -> str:
    # Same hybrid search + semantic reranking as Step 4
    ...

agent = Agent(
    client=chat_client,       # OpenAIChatCompletionClient (Azure OpenAI)
    name="HarryPotterRAG",
    instructions="...",
    tools=[search_books],
)

result = await agent.run("Who is Harry's godfather?")
```

You define the `search_books` function and mark it with `@tool`. The agent **decides on its own** when to call it. You never manually build prompts or stitch context — the agent handles the entire loop.

### What's better than Step 4

- **Autonomous tool calling** — The agent can reformulate queries, call the tool multiple times, or skip it entirely if the question doesn't need retrieval.
- **Easy to extend** — Adding a new tool is just writing another `@tool` function. No pipeline rewrite.
- **Cleaner code** — The `ask()` function from Step 4 (embed + search + format + call) is replaced by `agent.run(question)`.

---

## Step 6: Agent Framework + Foundry Client

Same architecture as Step 5, but swaps `OpenAIChatCompletionClient` for `FoundryChatClient`:

```python
from agent_framework.foundry import FoundryChatClient

chat_client = FoundryChatClient(
    project_endpoint=FOUNDRY_PROJECT_ENDPOINT,
    model=FOUNDRY_MODEL,
    credential=credential,
)
```

### What changes

| Aspect | Step 5 (OpenAI Client) | Step 6 (Foundry Client) |
|---|---|---|
| **Connection target** | Standalone Azure OpenAI resource | Azure AI Foundry project endpoint |
| **Model access** | Only models deployed in your AOAI resource | Any model deployed in your Foundry project (including non-OpenAI models) |
| **Identity** | Needs AOAI endpoint + deployment name | Needs Foundry project endpoint + model name |
| **Agent runtime** | Local | Local |

### Why use Foundry Client?

- **Unified project** — If your models, data, evaluations, and deployments all live in an Azure AI Foundry project, the Foundry client connects directly without needing separate AOAI resource configuration.
- **Model flexibility** — Foundry can host models from multiple providers. The same agent code works regardless of which model backs it.
- **Consistent identity** — One `DefaultAzureCredential` + project endpoint for everything.

The agent behavior is identical to Step 5 — it's purely a connectivity difference.

---

## Step 7: Foundry Agent Service (Server-Side Agent)

This is the biggest architectural shift. The agent definition **lives in the Foundry service**, not just in your notebook.

```
User Question
     │
     ▼
openai_client.responses.create()
     │
     ▼
┌──────────────────────────────────────────────┐
│              Foundry Agent Service            │
│                                               │
│  Agent: "HarryPotterRAG"                     │
│  ├── Instructions                            │
│  ├── Tools: [search_books]                   │
│  └── Session management (built-in Redis)     │
│                                               │
│  Agent decides: "call search_books"          │
│       │                                       │
│       ▼ function_call returned to client      │
└───────┬───────────────────────────────────────┘
        │
        ▼
Your code: execute search_books locally
        │
        ▼
Submit function output back to the service
        │
        ▼
Agent generates final answer
```

### How it works

```python
# Register the agent in Foundry (visible in the portal!)
agent = project.agents.create_version(
    agent_name="HarryPotterRAG",
    definition=PromptAgentDefinition(
        model=MODEL,
        instructions="...",
        tools=[search_tool],
    ),
)

# Query the agent via Responses API
response = openai_client.responses.create(
    input=question,
    extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
)
```

### Built-in Session Management (Redis-Backed)

This is the **killer feature** of Foundry Agent Service. In Step 4, every call is stateless. Steps 5–6 support multi-turn via `create_session()`, but the state lives in-memory and is lost when the process restarts. Foundry Agent Service moves that state server-side.

Foundry Agent Service provides **server-side session management** backed by built-in Redis:

```python
# Turn 1
answer, last_id = ask_agent("What is the Philosopher's Stone?")

# Turn 2 — automatic context from Turn 1, no manual history needed
answer, last_id = ask_agent("Who was trying to steal it and why?",
                            previous_response_id=last_id)
```

| Feature | Step 4 (Manual RAG) | Steps 5–6 (Agent Framework) | Step 7 (Foundry Service) |
|---|---|---|---|
| **Conversation history** | You manage it manually | SDK manages it via `create_session()` | Server manages it via `previous_response_id` — backed by Redis |
| **State persistence** | None | In-memory (lost when process restarts) | Server-side (persists across sessions, clients, restarts) |
| **Concurrent users** | You build the isolation | You build the isolation | Built-in session isolation per conversation |
| **Scaling** | You manage Redis/DB yourself | You manage Redis/DB yourself | Redis is built into the service |

### What's better than Steps 5–6

| ✅ | Benefit |
|---|---|
| **Portal visibility** | The agent shows up in the Azure AI Foundry portal — you can see its definition, test it, share it |
| **Server-side sessions** | Built-in Redis-backed conversation state — persists across restarts and clients (Steps 5–6 sessions are in-memory only) |
| **Multi-turn persistence** | Just pass `previous_response_id` — state survives restarts, unlike in-memory `create_session()` |
| **Versioning** | `create_version()` gives you version history of agent definitions |
| **Team collaboration** | Multiple developers / apps can reference the same registered agent |
| **Production-ready** | The agent definition is decoupled from the client — deploy once, consume from multiple apps |

---

## Head-to-Head Comparison

| Dimension | Step 4: Manual RAG | Step 5: Agent Framework (AOAI) | Step 6: Agent Framework (Foundry) | Step 7: Foundry Agent Service |
|---|---|---|---|---|
| **Architecture** | Procedural pipeline | Local autonomous agent | Local autonomous agent | Server-managed agent |
| **Tool calling** | None (hardcoded flow) | Agent decides autonomously | Agent decides autonomously | Agent decides autonomously |
| **Multi-turn** | ❌ Stateless | ✅ In-memory sessions (`create_session()`) | ✅ In-memory sessions (`create_session()`) | ✅ Server-side sessions (Redis) |
| **Conversation memory** | Manual | SDK-managed (in-memory) | SDK-managed (in-memory) | Automatic via `previous_response_id` (persistent) |
| **Portal visibility** | ❌ | ❌ | ❌ | ✅ Visible in Foundry portal |
| **Agent versioning** | ❌ | ❌ | ❌ | ✅ `create_version()` |
| **Extensibility** | Rewrite pipeline | Add `@tool` functions | Add `@tool` functions | Add tool schemas |
| **Model flexibility** | Azure OpenAI only | Azure OpenAI only | Any Foundry-hosted model | Any Foundry-hosted model |
| **Dependencies** | Minimal | `agent-framework` | `agent-framework-foundry` | `azure-ai-projects` |
| **Best for** | Prototyping, simple Q&A | Local dev, single-user | Local dev with Foundry models | Production, multi-user, multi-turn |

---

## When to Use What

### Use Step 4 (Manual RAG) when...
- You need a quick prototype or demo
- The workflow is simple and fixed (always search → answer)
- You don't need multi-turn conversations
- You want minimal dependencies

### Use Step 5/6 (Agent Framework) when...
- You want the agent to **decide** when and how to use tools
- You're adding multiple tools (search, calculator, database, API calls)
- You want clean, extensible code with `@tool` decorators
- You're developing locally and don't need server-side state

### Use Step 7 (Foundry Agent Service) when...
- You need **multi-turn conversations** with automatic state management
- Multiple users or applications will consume the same agent
- You want the agent **visible and manageable** in the Foundry portal
- You need **versioning** of agent definitions
- You're building a **production** system that needs session persistence (built-in Redis)
- Your team needs to collaborate on agent definitions

---

## The Progression

```
Step 4                    Step 5/6                    Step 7
Manual RAG         →    Agent Framework         →    Foundry Agent Service
─────────────────────────────────────────────────────────────────────────
Hardcoded pipeline       Autonomous tool use          Server-managed agent
No multi-turn            In-memory sessions           Persistent sessions (Redis)
Script-level             SDK-level                    Service-level
Single user              Single user                  Multi-user, multi-app
Prototype                Development                  Production
```

Each step adds a layer of capability without changing the core retrieval logic (hybrid search + semantic reranking stays the same). The difference is in **who orchestrates** the pipeline and **where state lives**.
