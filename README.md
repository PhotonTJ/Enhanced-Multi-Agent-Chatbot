Enhanced Multi-Agent Chatbot System Documentation
Table of Contents

Overview
Architecture
Installation & Setup
Core Components
Agent Personalities
Classification System
Memory System
Routing Logic
API Reference
Usage Examples
Configuration
Troubleshooting
Contributing

Overview
The Enhanced Multi-Agent Chatbot System is a sophisticated conversational AI platform that intelligently routes user messages to specialized agents based on multi-dimensional analysis. The system combines natural language processing, emotional intelligence, and context awareness to provide personalized responses.
Key Features

Multi-Agent Architecture: 5 specialized agents with unique personalities
Intelligent Classification: Multi-dimensional message analysis
Memory System: Persistent conversation context and user profiling
Crisis Intervention: Automatic detection and specialized support
Hybrid Intelligence: Context-aware routing with mood analysis
Session Management: Unique session tracking and conversation summaries

Architecture
User Input → Classifier → Intelligent Router → Specialized Agent → Response
     ↓              ↓            ↓                   ↓             ↓
  Session Init → Memory Update → Context Analysis → Personality → Summary
System Flow

Session Initialization: Creates unique session with memory context
Message Classification: Analyzes message across multiple dimensions
Intelligent Routing: Selects appropriate agent based on classification and context
Agent Processing: Specialized agent generates personalized response
Memory Update: Updates conversation history and user profile
Response Delivery: Returns enhanced response with personality

Installation & Setup
Prerequisites

Python 3.8+
GROQ API Key
Required packages (see requirements below)

Installation
bash# Clone or download the chatbot script
pip install langgraph langchain-groq pydantic typing-extensions

# Set your GROQ API key
export GROQ_API_KEY="your_groq_api_key_here"
Quick Start
pythonfrom enhanced_chatbot import run_enhanced_chatbot

# Run the interactive chatbot
run_enhanced_chatbot()
Core Components
1. State Management
pythonclass State(TypedDict):
    messages: Annotated[list, add_messages]
    classification: MessageClassifier | None
    memory: ConversationMemory
    context: dict
    session_id: str
    response_style: str
The State object maintains all conversation context, including message history, classification results, memory, and session information.
2. Message Classification
pythonclass MessageClassifier(BaseModel):
    message_type: MessageType           # Primary category
    urgency: Urgency                   # Urgency level
    topics: List[str]                  # Key topics (max 5)
    sentiment_score: float             # -1 to 1 sentiment
    confidence: float                  # Classification confidence
3. Memory System
pythonclass ConversationMemory(BaseModel):
    user_preferences: dict             # User preferences and settings
    conversation_history: List[dict]   # Complete conversation log
    user_mood_trend: List[float]       # Sentiment trend analysis
    topics_discussed: set              # All discussed topics
Agent Personalities
Dr. Empathy (Therapist Agent)

Role: Emotional support and therapeutic guidance
Style: Warm, understanding, uses reflective listening
Triggers: Emotional messages, crisis situations, declining mood trends
Specialties: Mental health support, emotional validation, coping strategies

Analytica (Logical Agent)

Role: Factual analysis and logical reasoning
Style: Precise, methodical, data-driven
Triggers: Logical questions, analytical requests, factual inquiries
Specialties: Problem-solving, data analysis, step-by-step reasoning

Artisan (Creative Agent)

Role: Creative ideation and artistic guidance
Style: Imaginative, inspiring, uses metaphors
Triggers: Creative requests, brainstorming, artistic projects
Specialties: Creative writing, design ideas, artistic inspiration

CodeMaster (Technical Agent)

Role: Technical assistance and programming help
Style: Detailed, systematic, solution-focused
Triggers: Technical questions, programming issues, system problems
Specialties: Code debugging, system architecture, technical explanations

Buddy (Casual Agent)

Role: General conversation and friendly chat
Style: Friendly, relaxed, conversational
Triggers: Casual messages, general inquiries, social interaction
Specialties: General knowledge, friendly conversation, social support

Classification System
Message Types
TypeDescriptionExample TriggersEMOTIONALFeelings, emotions, personal struggles"I'm feeling sad", "I need support"LOGICALFacts, analysis, reasoning"How does this work?", "Analyze this data"CREATIVEIdeas, brainstorming, artistic content"Help me write a story", "Design ideas"TECHNICALCode, systems, technical problems"Debug this code", "System architecture"CASUALGeneral chat, social interaction"Hello", "What's up?", "Random question"
Urgency Levels
LevelDescriptionResponse StrategyCRISISImmediate intervention neededCrisis therapist, emergency resourcesHIGHImportant, time-sensitiveDirect, actionable responsesMEDIUMNotable concernBalanced approachLOWGeneral inquiryConversational, detailed responses
Sentiment Scoring

Range: -1.0 (very negative) to +1.0 (very positive)
Usage: Mood tracking, context adjustment, agent selection
Threshold: < -0.5 triggers therapeutic support consideration

Memory System
Conversation History
python{
    "timestamp": "2024-01-15T10:30:00",
    "message": "User message content",
    "classification": {
        "message_type": "emotional",
        "urgency": "medium",
        "sentiment_score": -0.3
    }
}
Mood Trend Analysis

Tracks sentiment scores over time
Calculates rolling averages for context
Triggers therapeutic support for declining trends
Influences agent selection and response style

Topic Memory

Maintains set of all discussed topics
Enables contextual references in responses
Helps with conversation continuity
Supports personalized interactions

Routing Logic
Primary Routing
pythonroute_map = {
    MessageType.EMOTIONAL: "therapist",
    MessageType.LOGICAL: "logical",
    MessageType.CREATIVE: "creative",
    MessageType.TECHNICAL: "technical",
    MessageType.CASUAL: "casual"
}
Context-Aware Adjustments
Crisis Detection

Urgency level CRISIS → Crisis Therapist
Provides immediate resources and professional referrals

Mood-Based Routing

Average mood < -0.5 over last 3 messages → Hybrid Therapist
Combines therapeutic support with requested functionality
Maintains empathy while addressing original intent

Confidence-Based Fallbacks

Low classification confidence → Casual Agent
Provides general support while gathering more context

API Reference
Core Functions
initialize_session(state: State) -> dict
Initializes a new conversation session with unique ID and memory.
Returns:

session_id: Unique session identifier
memory: Fresh ConversationMemory instance
context: Session metadata

enhanced_classify_message(state: State) -> dict
Performs multi-dimensional message analysis.
Parameters:

state: Current conversation state

Returns:

classification: MessageClassifier instance
memory: Updated memory with classification data

intelligent_router(state: State) -> dict
Routes messages to appropriate agents based on classification and context.
Parameters:

state: Current conversation state with classification

Returns:

next: Target agent identifier

create_enhanced_agent(agent_type: str, personality: dict) -> function
Factory function for creating specialized agents.
Parameters:

agent_type: Agent identifier
personality: Personality configuration

Returns:

Agent function with embedded personality

Agent Functions
All agent functions follow the same signature:
pythondef agent_function(state: State) -> dict:
    """
    Process user message with agent-specific personality and expertise.
    
    Parameters:
        state: Current conversation state
        
    Returns:
        dict: Updated messages with agent response
    """
Specialized Agents
crisis_therapist_agent(state: State) -> dict
Handles crisis situations with immediate support and resources.
hybrid_therapist_agent(state: State) -> dict
Combines therapeutic support with task-specific assistance.
conversation_summarizer(state: State) -> State
Generates conversation insights and session summary.
Usage Examples
Basic Usage
pythonfrom enhanced_chatbot import build_enhanced_graph

# Create the chatbot graph
graph = build_enhanced_graph()

# Process a single message
state = graph.invoke({
    "messages": [{"role": "user", "content": "I'm feeling overwhelmed"}],
    "classification": None,
    "memory": ConversationMemory(),
    "context": {},
    "session_id": "",
    "response_style": "adaptive"
})

print(state["messages"][-1]["content"])
Interactive Session
pythonfrom enhanced_chatbot import run_enhanced_chatbot

# Start interactive conversation
run_enhanced_chatbot()
Custom Agent Creation
pythonfrom enhanced_chatbot import create_enhanced_agent, AgentPersonalities

# Create custom agent
custom_personality = {
    "name": "DataBot",
    "style": "analytical, data-focused, uses statistics",
    "greeting": "Let's dive into the data! 📊"
}

data_agent = create_enhanced_agent("data", custom_personality)
Accessing Conversation Memory
python# After processing messages
memory = state["memory"]

# Get mood trend
average_mood = sum(memory.user_mood_trend) / len(memory.user_mood_trend)

# Get recent topics
recent_topics = list(memory.topics_discussed)[-5:]

# Get conversation history
last_interactions = memory.conversation_history[-3:]
Configuration
Environment Variables
bash# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
CHATBOT_LOG_LEVEL=INFO
CHATBOT_MAX_HISTORY=100
CHATBOT_SESSION_TIMEOUT=3600
Model Configuration
python# Default configuration
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.7
)

# Custom configuration
llm = ChatGroq(
    model="llama3-70b-8192",  # Larger model for better performance
    temperature=0.5,           # Lower temperature for more consistent responses
    max_tokens=1000           # Limit response length
)
Agent Personality Customization
python# Modify existing personalities
AgentPersonalities.THERAPIST["style"] = "warm, professional, solution-focused"
AgentPersonalities.THERAPIST["greeting"] = "I'm here to help you navigate this. 🌟"

# Add new personality
AgentPersonalities.RESEARCHER = {
    "name": "Scholar",
    "style": "academic, thorough, evidence-based",
    "greeting": "Let's explore this topic in depth! 🔍"
}
Troubleshooting
Common Issues
API Key Errors
Error: Invalid API key
Solution: Verify GROQ_API_KEY is set correctly
Classification Failures
Error: Classification returned None
Solution: Check message content is not empty, verify LLM connection
Memory Overflow
Error: Memory usage too high
Solution: Implement memory cleanup, limit conversation history
Agent Routing Issues
Error: Unknown agent type
Solution: Verify agent exists in routing map, check classification output
Debug Mode
Enable detailed logging:
pythonimport logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
run_enhanced_chatbot()
Performance Optimization
Memory Management
python# Limit conversation history
MAX_HISTORY = 50

def cleanup_memory(memory: ConversationMemory):
    if len(memory.conversation_history) > MAX_HISTORY:
        memory.conversation_history = memory.conversation_history[-MAX_HISTORY:]
    
    if len(memory.user_mood_trend) > MAX_HISTORY:
        memory.user_mood_trend = memory.user_mood_trend[-MAX_HISTORY:]
Response Caching
pythonfrom functools import lru_cache

@lru_cache(maxsize=128)
def cached_classification(message_content: str):
    # Cache classification results for identical messages
    pass
Advanced Features
Multi-Session Support
pythonclass SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id: str) -> State:
        return self.sessions.get(session_id)
    
    def create_session(self) -> str:
        session_id = generate_session_id()
        self.sessions[session_id] = initialize_state()
        return session_id
Custom Classification Models
pythondef custom_classifier(message: str) -> MessageClassifier:
    # Implement custom classification logic
    # Could use different models, rule-based systems, etc.
    pass
Webhook Integration
pythonfrom flask import Flask, request, jsonify

app = Flask(__name__)
graph = build_enhanced_graph()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    state = graph.invoke(data)
    return jsonify(state["messages"][-1])
Contributing
Development Setup
bash# Clone repository
git clone <repository-url>
cd enhanced-chatbot

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
Adding New Agents

Define personality in AgentPersonalities
Create agent function using create_enhanced_agent
Add routing logic to intelligent_router
Add agent to graph in build_enhanced_graph
Write tests for new functionality

Testing
python# Test message classification
def test_classification():
    state = {"messages": [{"role": "user", "content": "I need help with Python"}]}
    result = enhanced_classify_message(state)
    assert result["classification"].message_type == MessageType.TECHNICAL

# Test agent responses
def test_agent_response():
    state = create_test_state("Hello there!")
    response = casual_agent(state)
    assert "Buddy" in response["messages"][0]["content"]
Code Style Guidelines

Follow PEP 8 for Python code style
Use type hints for all function parameters and returns
Document all public functions with docstrings
Keep functions focused and under 50 lines
Use meaningful variable names
Add comments for complex logic
