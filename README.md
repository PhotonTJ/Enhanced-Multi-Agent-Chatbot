# Enhanced Multi-Agent Chatbot System

## 📚 Table of Contents

* [Overview](#overview)
* [Architecture](#architecture)
* [Installation & Setup](#installation--setup)
* [Core Components](#core-components)
* [Agent Personalities](#agent-personalities)
* [Classification System](#classification-system)
* [Memory System](#memory-system)
* [Routing Logic](#routing-logic)
* [API Reference](#api-reference)
* [Usage Examples](#usage-examples)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)

---

## 🌐 Overview

The **Enhanced Multi-Agent Chatbot System** is an advanced conversational AI platform that intelligently routes user inputs to specialized agents based on deep contextual and emotional analysis.

### Key Features

* ✨ **Multi-Agent Architecture**: Five domain-specific agents with distinct personalities
* 🤔 **Intelligent Classification**: Multi-dimensional message analysis
* 📂 **Memory System**: Persistent memory of user preferences and conversation history
* ⚠️ **Crisis Intervention**: Automatic detection and support for critical messages
* 🧐 **Hybrid Intelligence**: Combines sentiment, context, and classification for routing
* 🔍 **Session Management**: Unique session tracking with summaries

---

## 💡 Architecture

```text
User Input → Classifier → Intelligent Router → Specialized Agent → Response
     ↓              ↓            ↓                   ↓             ↓
  Session Init → Memory Update → Context Analysis → Personality → Summary
```

### Process Breakdown

* **Session Initialization**: Generates unique session ID and context
* **Message Classification**: Analyzes sentiment, urgency, and topics
* **Intelligent Routing**: Chooses agent based on mood, topic, and context
* **Agent Processing**: Generates a personalized response
* **Memory Update**: Updates context, history, and mood trends
* **Response Delivery**: Enhanced and stylized output

---

## 📁 Installation & Setup

### Prerequisites

* Python 3.8+
* GROQ API Key

### Install Required Packages

```bash
pip install langgraph langchain-groq pydantic typing-extensions
```

### Set Your API Key

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### Quick Start

```python
from enhanced_chatbot import run_enhanced_chatbot
run_enhanced_chatbot()
```

---

## 📊 Core Components

### 1. State Management

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    classification: MessageClassifier | None
    memory: ConversationMemory
    context: dict
    session_id: str
    response_style: str
```

### 2. Message Classification

```python
class MessageClassifier(BaseModel):
    message_type: MessageType
    urgency: Urgency
    topics: List[str]
    sentiment_score: float
    confidence: float
```

### 3. Memory System

```python
class ConversationMemory(BaseModel):
    user_preferences: dict
    conversation_history: List[dict]
    user_mood_trend: List[float]
    topics_discussed: set
```

---

## 🧩 Agent Personalities

### 👩‍🎓 Dr. Empathy (Therapist)

* Role: Emotional support
* Style: Warm, understanding
* Triggers: Crisis, declining mood

### 🔢 Analytica (Logical)

* Role: Factual reasoning
* Style: Precise, methodical
* Triggers: Logical questions

### 🎨 Artisan (Creative)

* Role: Artistic inspiration
* Style: Imaginative, metaphorical
* Triggers: Creative tasks

### 💻 CodeMaster (Technical)

* Role: Coding and tech help
* Style: Detailed, solution-focused
* Triggers: Debugging, architecture

### 🤝 Buddy (Casual)

* Role: Social interaction
* Style: Friendly, informal
* Triggers: Casual chats

---

## 📈 Classification System

### Message Types

| Type      | Description                     | Example                  |
| --------- | ------------------------------- | ------------------------ |
| EMOTIONAL | Emotions, personal struggles    | "I'm feeling lost"       |
| LOGICAL   | Facts, analysis, reasoning      | "Explain this algorithm" |
| CREATIVE  | Ideas, brainstorming            | "Help write a poem"      |
| TECHNICAL | Programming or systems help     | "Fix this code"          |
| CASUAL    | Small talk, general interaction | "Hey, how are you?"      |

### Urgency Levels

| Level  | Description           | Strategy                        |
| ------ | --------------------- | ------------------------------- |
| CRISIS | Immediate help needed | Crisis agent + emergency advice |
| HIGH   | Time-sensitive        | Fast, clear actionable replies  |
| MEDIUM | Concerned, not urgent | Balanced, thoughtful response   |
| LOW    | General inquiry       | Friendly and descriptive        |

### Sentiment Scoring

* **Range**: -1.0 (very negative) to +1.0 (very positive)
* **Use**: Routing, personalization, and mood detection
* **Threshold**: Avg < -0.5 triggers hybrid therapeutic agent

---

## 📂 Memory System

### Example History Entry

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "message": "User message content",
  "classification": {
    "message_type": "emotional",
    "urgency": "medium",
    "sentiment_score": -0.3
  }
}
```

### Features

* Tracks **sentiment trends** for emotional decline
* Maintains **topics** to improve continuity
* Stores **user preferences** for personalization

---

## 🛠️ Routing Logic

### Primary Routing Map

```python
route_map = {
    MessageType.EMOTIONAL: "therapist",
    MessageType.LOGICAL: "logical",
    MessageType.CREATIVE: "creative",
    MessageType.TECHNICAL: "technical",
    MessageType.CASUAL: "casual"
}
```

### Context-Aware Adjustments

* **Crisis → Crisis Therapist**
* **Negative Mood Trend → Hybrid Therapist**
* **Low Confidence → Casual Agent**

---

## 📲 API Reference

### Core Functions

```python
def initialize_session(state: State) -> dict
```

Initializes a new session and returns session data.

```python
def enhanced_classify_message(state: State) -> dict
```

Returns `MessageClassifier` object and updates memory.

```python
def intelligent_router(state: State) -> dict
```

Determines agent to route the message to.

```python
def create_enhanced_agent(agent_type: str, personality: dict) -> function
```

Returns an agent function with embedded personality.

### Agent Interface

```python
def agent_function(state: State) -> dict:
    """
    Processes message with specialized response.
    """
```

---

## 🔄 Usage Examples

### Basic Usage

```python
from enhanced_chatbot import build_enhanced_graph
state = build_enhanced_graph().invoke({
    "messages": [{"role": "user", "content": "I'm feeling overwhelmed"}],
    "classification": None,
    "memory": ConversationMemory(),
    "context": {},
    "session_id": "",
    "response_style": "adaptive"
})
print(state["messages"][-1]["content"])
```

### Interactive Mode

```python
from enhanced_chatbot import run_enhanced_chatbot
run_enhanced_chatbot()
```

---

## 🔧 Configuration

### Environment Variables

```bash
GROQ_API_KEY=your_groq_api_key_here
CHATBOT_LOG_LEVEL=INFO
CHATBOT_MAX_HISTORY=100
CHATBOT_SESSION_TIMEOUT=3600
```

### Model Settings

```python
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)
```

---

## 🚫 Troubleshooting

| Issue                | Solution                                  |
| -------------------- | ----------------------------------------- |
| Invalid API key      | Check `GROQ_API_KEY` environment variable |
| Classification fails | Check message content / model connection  |
| Memory overflow      | Implement cleanup, reduce history size    |
| Unknown agent type   | Check classification and routing logic    |

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🚀 Contributing

### Setup

```bash
git clone <repo-url>
cd enhanced-chatbot
pip install -r requirements-dev.txt
pytest tests/
flake8 src/
black src/
```

### Add a New Agent

1. Define in `AgentPersonalities`
2. Create via `create_enhanced_agent`
3. Add to `intelligent_router`
4. Register in `build_enhanced_graph`

### Tests

```python
def test_classification():
    ...

def test_agent_response():
    ...
```

### Code Style

* Follow **PEP 8**
* Use **type hints**
* Add **docstrings** and **comments**
* Use **meaningful variable names**

---

Feel free to contribute ideas, fixes, or enhancements via pull requests! 🚀
