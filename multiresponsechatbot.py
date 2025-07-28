from typing import Annotated, Literal, Optional, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
import os
import random
from enum import Enum

os.environ["GROQ_API_KEY"] = ""


llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)

class MessageType(str, Enum):
    EMOTIONAL = "emotional"
    LOGICAL = "logical"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    CASUAL = "casual"

class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRISIS = "crisis"

class MessageClassifier(BaseModel):
    message_type: MessageType = Field(description="Primary category of the message")
    urgency: Urgency = Field(description="Urgency level of the message")
    topics: List[str] = Field(description="Key topics mentioned", max_items=5)
    sentiment_score: float = Field(description="Sentiment from -1 (negative) to 1 (positive)", ge=-1, le=1)
    confidence: float = Field(description="Confidence in classification", ge=0, le=1)

class ConversationMemory(BaseModel):
    user_preferences: dict = Field(default_factory=dict)
    conversation_history: List[dict] = Field(default_factory=list)
    user_mood_trend: List[float] = Field(default_factory=list)
    topics_discussed: set = Field(default_factory=set)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    classification: Optional[MessageClassifier]
    memory: ConversationMemory
    context: dict
    session_id: str

# Agent personalities
class AgentPersonalities:
    THERAPIST = {
        "name": "Dr. Empathy",
        "style": "warm, understanding, uses reflective listening",
        "greeting": "I'm here to listen and support you. 💙"
    }
    
    LOGICAL = {
        "name": "Analytica",
        "style": "precise, methodical, data-driven",
        "greeting": "Let's analyze this step by step. 🧠"
    }
    
    CREATIVE = {
        "name": "Artisan",
        "style": "imaginative, inspiring, uses metaphors",
        "greeting": "Let's explore the creative possibilities! ✨"
    }
    
    TECHNICAL = {
        "name": "CodeMaster",
        "style": "detailed, systematic, solution-focused",
        "greeting": "Time to dive into the technical details! ⚙️"
    }
    
    CASUAL = {
        "name": "Buddy",
        "style": "friendly, relaxed, conversational",
        "greeting": "Hey there! What's on your mind? 😊"
    }

def initialize_session(state: State):
    """Initialize conversation session with memory and context"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
    return {
        "session_id": session_id,
        "memory": ConversationMemory(),
        "context": {"start_time": datetime.now().isoformat()}
    }

def enhanced_classify_message(state: State):
    """Enhanced message classification with multiple dimensions"""
    last_message = state["messages"][-1]
    
    classifier_llm = llm.with_structured_output(MessageClassifier)
    
    classification_prompt = """
    Analyze this message across multiple dimensions:
    1. Primary type: emotional (feelings/support), logical (facts/analysis), creative (ideas/brainstorming), technical (code/systems), casual (chat/social)
    2. Urgency: crisis (immediate help needed), high (important/time-sensitive), medium (notable concern), low (general inquiry)
    3. Key topics mentioned (up to 5)
    4. Sentiment score from -1 (very negative) to 1 (very positive)
    5. Your confidence in this classification (0-1)
    
    Be thorough but concise in your analysis.
    """
    
    result = classifier_llm.invoke([
        {"role": "system", "content": classification_prompt},
        {"role": "user", "content": last_message.content}
    ])
    
    # Update memory with classification
    memory = state.get("memory", ConversationMemory())
    memory.conversation_history.append({
        "timestamp": datetime.now().isoformat(),
        "message": last_message.content,
        "classification": result.model_dump()
    })
    memory.user_mood_trend.append(result.sentiment_score)
    memory.topics_discussed.update(result.topics)
    
    print(f"🎯 [CLASSIFICATION] Type: {result.message_type.value.upper()}")
    print(f"⚡ [URGENCY] Level: {result.urgency.value.upper()}")
    print(f"📊 [SENTIMENT] Score: {result.sentiment_score:.2f}")
    print(f"🎲 [CONFIDENCE] {result.confidence:.2f}")
    print(f"🏷️ [TOPICS] {', '.join(result.topics)}")
    
    return {"classification": result, "memory": memory}

def intelligent_router(state: State):
    """Smart routing with context awareness"""
    classification = state.get("classification")
    memory = state.get("memory", ConversationMemory())
    
    if not classification:
        return "casual"
    
    # Crisis intervention routing
    if classification.urgency == Urgency.CRISIS:
        print("🚨 [ROUTER] Crisis detected - routing to crisis therapist")
        return "crisis_therapist"
    
    # Primary routing logic
    route_map = {
        MessageType.EMOTIONAL: "therapist",
        MessageType.LOGICAL: "logical",
        MessageType.CREATIVE: "creative",
        MessageType.TECHNICAL: "technical",
        MessageType.CASUAL: "casual"
    }
    
    selected_route = route_map.get(classification.message_type, "casual")
    
    # Context-aware adjustments
    if len(memory.user_mood_trend) > 3:
        avg_mood = sum(memory.user_mood_trend[-3:]) / 3
        if avg_mood < -0.5 and selected_route != "therapist":
            print("📉 [ROUTER] Detected declining mood - adding therapeutic support")
            selected_route = "hybrid_therapist"
    
    print(f"🧭 [ROUTER] Directing to: {selected_route.upper()}")
    return selected_route

def create_enhanced_agent(agent_type: str, personality: dict):
    """Factory function for creating enhanced agents"""
    def agent_function(state: State):
        last_message = state["messages"][-1]
        classification = state.get("classification")
        memory = state.get("memory", ConversationMemory())
        
        # Build context-aware system prompt
        base_prompt = f"""
        You are {personality['name']}, an AI assistant with this personality: {personality['style']}.
        
        Current message classification:
        - Type: {classification.message_type.value if classification else 'unknown'}
        - Urgency: {classification.urgency.value if classification else 'medium'}
        - Sentiment: {classification.sentiment_score if classification else 0}
        - Topics: {', '.join(classification.topics) if classification else 'general'}
        
        Recent conversation context:
        - Average mood trend: {sum(memory.user_mood_trend[-3:]) / max(1, len(memory.user_mood_trend[-3:])) if memory.user_mood_trend else 0:.2f}
        - Previous topics: {', '.join(list(memory.topics_discussed)[-5:]) if memory.topics_discussed else 'none'}
        
        Guidelines:
        1. Maintain your personality while being helpful
        2. Reference previous context when relevant
        3. Adapt your response length to the urgency level
        4. Use appropriate emotional intelligence
        """
        
        # Urgency-based response modifications
        if classification and classification.urgency == Urgency.HIGH:
            base_prompt += "\n5. This is high urgency - be direct and actionable"
        elif classification and classification.urgency == Urgency.LOW:
            base_prompt += "\n5. This is low urgency - feel free to be more conversational"
        
        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": last_message.content}
        ]
        
        reply = llm.invoke(messages)
        
        # Add personality flourish
        response_with_personality = f"{personality['greeting']}\n\n{reply.content}"
        
        return {"messages": [{"role": "assistant", "content": response_with_personality}]}
    
    return agent_function

# Create specialized agents
therapist_agent = create_enhanced_agent("therapist", AgentPersonalities.THERAPIST)
logical_agent = create_enhanced_agent("logical", AgentPersonalities.LOGICAL)
creative_agent = create_enhanced_agent("creative", AgentPersonalities.CREATIVE)
technical_agent = create_enhanced_agent("technical", AgentPersonalities.TECHNICAL)
casual_agent = create_enhanced_agent("casual", AgentPersonalities.CASUAL)

def crisis_therapist_agent(state: State):
    """Specialized crisis intervention agent"""
    last_message = state["messages"][-1]
    
    crisis_prompt = """
    You are a crisis intervention specialist. The user may be in distress.
    
    Your response should:
    1. Acknowledge their feelings with empathy
    2. Provide immediate emotional support
    3. Suggest helpful resources if appropriate
    4. Encourage professional help for serious situations
    5. Be calm, supportive, and non-judgmental
    
    Remember: You're an AI and cannot replace professional crisis intervention.
    """
    
    messages = [
        {"role": "system", "content": crisis_prompt},
        {"role": "user", "content": last_message.content}
    ]
    
    reply = llm.invoke(messages)
    
    crisis_resources = """
    
    🆘 **Immediate Resources:**
    • Crisis Text Line: Text HOME to 741741
    • National Suicide Prevention Lifeline: 988
    • Emergency Services: 911
    """
    
    return {"messages": [{"role": "assistant", "content": f"🚨 **Crisis Support** 🚨\n\n{reply.content}{crisis_resources}"}]}

def hybrid_therapist_agent(state: State):
    """Hybrid agent that combines therapeutic support with the original intent"""
    last_message = state["messages"][-1]
    classification = state.get("classification")
    
    hybrid_prompt = f"""
    You are a supportive AI that combines therapeutic empathy with {classification.message_type.value if classification else 'general'} assistance.
    
    First, acknowledge any emotional aspects, then provide the requested help.
    Be warm but also practically helpful.
    """
    
    messages = [
        {"role": "system", "content": hybrid_prompt},
        {"role": "user", "content": last_message.content}
    ]
    
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": f"💙 **Supportive Assistant** 💙\n\n{reply.content}"}]}

def conversation_summarizer(state: State):
    """Generate conversation summary and insights"""
    memory = state.get("memory", ConversationMemory())
    
    if len(memory.conversation_history) > 1:
        summary = f"""
        📋 **Conversation Summary:**
        • Messages exchanged: {len(memory.conversation_history)}
        • Topics covered: {', '.join(list(memory.topics_discussed))}
        • Mood trend: {sum(memory.user_mood_trend) / len(memory.user_mood_trend):.2f if memory.user_mood_trend else 0:.2f}
        • Session ID: {state.get('session_id', 'unknown')}
        """
        print(summary)
    
    return state

# Build the enhanced graph
def build_enhanced_graph():
    graph_builder = StateGraph(State)
    
    # Add all nodes
    graph_builder.add_node("initialize", initialize_session)
    graph_builder.add_node("classifier", enhanced_classify_message)
    graph_builder.add_node("therapist", therapist_agent)
    graph_builder.add_node("logical", logical_agent)
    graph_builder.add_node("creative", creative_agent)
    graph_builder.add_node("technical", technical_agent)
    graph_builder.add_node("casual", casual_agent)
    graph_builder.add_node("crisis_therapist", crisis_therapist_agent)
    graph_builder.add_node("hybrid_therapist", hybrid_therapist_agent)
    graph_builder.add_node("summarizer", conversation_summarizer)
    
    # Build the flow
    graph_builder.add_edge(START, "initialize")
    graph_builder.add_edge("initialize", "classifier")
    graph_builder.add_conditional_edges("classifier", intelligent_router)
    
    # All agents end at summarizer
    for agent in ["therapist", "logical", "creative", "technical", "casual", "crisis_therapist", "hybrid_therapist"]:
        graph_builder.add_edge(agent, "summarizer")
    
    graph_builder.add_edge("summarizer", END)
    
    return graph_builder.compile()

# Enhanced conversation loop
def run_enhanced_chatbot():
    graph = build_enhanced_graph()
    
    print("🤖 **Enhanced Multi-Agent Chatbot** 🚀")
    print("=" * 50)
    print("Available agents: Dr. Empathy, Analytica, Artisan, CodeMaster, Buddy")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 50)
    
    while True:
        user_input = input("\n💬 You: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Thanks for chatting! Take care!")
            break
        elif user_input.lower() == 'help':
            print("""
            🆘 **Available Commands:**
            • 'quit' - Exit the chatbot
            • 'help' - Show this help message
            • Just type normally to chat with the AI agents!
            
            The system will automatically route you to the most appropriate agent based on your message.
            """)
            continue
        elif not user_input:
            continue
        
        try:
            state = graph.invoke({
                "messages": [{"role": "user", "content": user_input}],
                "classification": None,
                "memory": ConversationMemory(),
                "context": {},
                "session_id": ""
            })
            
            print(f"\n🤖 Bot: {state['messages'][-1].content}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again or type 'help' for assistance.")

if __name__ == "__main__":
    run_enhanced_chatbot()