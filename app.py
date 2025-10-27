"""
Discussion Partner Chatbot - Web Interface with Voice Recording
AI-based DDL for Teaching Disagreement Pragmatics
Streamlit Version - Includes voice recording for Activity 3
REVISED VERSION: INCLUDES implicit feedback + both low and high power debates
"""

import streamlit as st
from openai import OpenAI
import json
import datetime
import io
from typing import Dict, List, Optional
from audio_recorder_streamlit import audio_recorder

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Discussion Partner",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS (keeping original)
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .activity-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff7f0e;
    }
    .dialogue-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
    }
    .corpus-example {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2ca02c;
        margin: 0.5rem 0;
        font-style: italic;
    }
    .scenario-box {
        background-color: #ffe6e6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #d62728;
        margin: 1rem 0;
    }
    .context-reminder {
        background-color: #f3e5f5;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #9c27b0;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .chat-message-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .chat-message-assistant {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #757575;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #145a8a;
        transform: scale(1.05);
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .voice-recording-box {
        background-color: #fff9c4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #fbc02d;
        margin: 1rem 0;
    }
    .feedback-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "gpt-4"
TEMPERATURE = 0.7
MAX_TOKENS = 500

# ============================================================================
# SYSTEM PROMPT - WITH ENHANCED IMPLICIT FEEDBACK
# ============================================================================

SYSTEM_PROMPT = """You are "Discussion Partner," an AI having a genuine conversation with a language learner.

🎯 YOUR PRIMARY GOAL: Have an AUTHENTIC, ENGAGING conversation while giving NATURAL FEEDBACK on register appropriateness.

CRITICAL PRINCIPLES (in order of importance):

1. **RESPOND TO THEIR IDEAS FIRST** 
   - React authentically to what they actually said
   - Engage with their content, arguments, and reasoning
   - Ask follow-up questions naturally
   - Show genuine interest in their perspective
   - Build on their points before disagreeing

2. **MODEL APPROPRIATE LANGUAGE NATURALLY**
   - Use disagreement patterns naturally in YOUR responses (never tell them to use patterns)
   - For friends/classmates: "Yeah but...", "I see your point but...", "I get that, but..."
   - For boss/authority: "I understand your concern, however...", "I can see where you're coming from, but perhaps..."
   - Let them learn by seeing you use these patterns, not by being told

3. **GIVE IMPLICIT FEEDBACK CONVERSATIONALLY** ⭐ IMPORTANT
   - Actively monitor if their register matches the relationship
   - If their language is inappropriate for the relationship, react naturally as that person would
   
   **Too casual/direct with boss:**
   - "Whoa, that's quite direct... As your boss, I'd appreciate a more diplomatic approach here."
   - "That's a bit informal for a professional conversation. Let me remind you this is a workplace discussion."
   - "I appreciate your honesty, but that tone isn't appropriate for an employee speaking to their supervisor."
   
   **Too formal/indirect with friend:**
   - "Haha, why so formal? Come on, we're friends! Just tell me what you really think."
   - "You sound like you're giving a presentation! Relax, it's just me."
   - "Dude, you don't need to be so polite. We're buddies, speak naturally!"
   
   **Appropriate register:**
   - Just continue naturally without commenting on their language
   - With boss: "I appreciate you bringing this up" / "Thank you for sharing your perspective"
   - With friend: "Yeah, I hear you" / "Fair point, man"

4. **MAINTAIN RELATIONSHIP CONTEXT**
   - Occasionally reference the relationship naturally: "Look, as your friend...", "From a management perspective...", "Come on, buddy..."
   - Make it feel like a real conversation with that person

5. **NEVER EXPLICITLY TEACH GRAMMAR**
   - Don't say: "The correct pattern is...", "You should use...", "Try saying..."
   - Don't break the fourth wall to teach linguistic features
   - Give feedback on APPROPRIATENESS and TONE, not grammar rules

CONVERSATION STYLE BY RELATIONSHIP:

**With Friends/Classmates (Low Power):**
- Be casual, direct, energetic
- Use contractions freely: "don't", "I'm", "you're"
- Start with "Yeah but...", "I know, but...", "True, but..."
- Keep responses shorter and punchier
- Show emotion: "Come on!", "Really?", "No way!"
- Example: "Yeah but don't you think social media also helps people stay connected? I mean, I talk to my friends way more now than before."

**With Boss/Authority (High Power):**
- Be professional, measured, diplomatic
- Use more formal language: "I understand", "Perhaps", "I was wondering"
- Acknowledge before disagreeing: "I see your point, however..."
- Keep responses longer and more elaborate
- Stay respectful but firm
- Example: "I understand your concern about the schedule. However, I was wondering if we could discuss alternative arrangements, as I have classes in the morning that I can't miss."

RESPONDING TO STUDENT INPUT:

Step 1: Evaluate their register
- Does it match the relationship (casual for friends, formal for boss)?
- Is it too direct or too indirect for the context?

Step 2: Engage with their CONTENT
- "That's an interesting point about..."
- "I see what you're saying about..."

Step 3: Give implicit feedback if register is notably off
- React naturally as that person would
- Don't stop the conversation, but signal the mismatch

Step 4: Model disagreement naturally while continuing the debate
- Friends: "Yeah but what about..." / "I get that, but..."
- Boss: "I understand your perspective, though I wonder if..." / "That's a fair point, however..."

Step 5: Continue the conversation
- Ask follow-up questions
- Introduce new angles
- Keep the debate/discussion flowing

Remember: You're a REAL PERSON first, but also a teacher helping them learn appropriate register. Balance authenticity with pedagogical feedback!"""

DIALOGUES = {
    "mobile_phones": {
        "title": "Mobile Phones on Trains",
        "context": "Eden (boss) and Tiara (employee) are talking",
        "power": "high",
        "dialogue": """**Eden:** Do you think that mobile phones should be banned on trains? Should they be forbidden?

**Tiara:** Because people speak too loudly?

**Eden:** Yes. Some people find that very annoying.

**Tiara:** I can see their point. It is sometimes annoying. But I don't agree that they should be banned."""
    },
    "life_expectancy": {
        "title": "Life Expectancy",
        "context": "Linda and Semih are friends discussing how long people will live",
        "power": "low",
        "dialogue": """**Linda:** Scientists say people will live over 100 years. I'm not convinced this is good.

**Semih:** Why aren't you keen on people living over 100?

**Linda:** When you look at 100-year-old people, they're not in excellent physical condition.

**Semih:** Well I agree but medicines and scientific research has been progressing. Maybe there are some kind of medicines in the future which can help.

**Linda:** Yes but if people live over 100 and retire later, then young people can't find jobs because older people keep working.

**Semih:** Well I agree but maybe we can develop more jobs..."""
    }
}

CORPUS_EXAMPLES = {
    "high_power": [
        "I can see their point. It is sometimes annoying. But I don't agree that they should be banned.",
        "I can understand your opinion erm but I was still wondering...",
        "I agree with this point but don't you think maybe the fact that times are changing is a good thing?",
        "I understand his situation but I'm not sure if I should do it"
    ],
    "low_power": [
        "Yeah but there are some disadvantages like er...",
        "yeah I agree but I still the problem is that...",
        "Well I agree but maybe we can develop more jobs",
        "Yes but if people are going to live over a hundred and they're probably going to retire later..."
    ]
}

# ============================================================================
# DEBATE_TOPICS - NOW WITH 2 LOW-POWER + 2 HIGH-POWER
# ============================================================================

DEBATE_TOPICS = [
    {
        "id": "social_media",
        "topic": "Social Media",
        "power": "low",
        "ai_position": "Social media is helpful",
        "ai_opening": "Hey! So you think social media is harmful? Yes, I know it can cause some problems, but I think it really helps people stay connected with friends and family.",
        "corpus_patterns": "low_power",
        "relationship": "friends"
    },
    {
        "id": "homework",
        "topic": "Homework",
        "power": "low",
        "ai_position": "Homework is necessary",
        "ai_opening": "Alright, homework debate! Yeah, I understand homework can be boring, but I think it's really important for learning. Don't you think practice helps?",
        "corpus_patterns": "low_power",
        "relationship": "classmates"
    },
    {
        "id": "dress_code",
        "topic": "Workplace Dress Code",
        "power": "high",
        "ai_position": "Professional dress code is necessary",
        "ai_opening": "I understand you have concerns about the dress code policy. However, I believe maintaining professional attire is important for our company image and client relationships. Could you share your perspective on this?",
        "corpus_patterns": "high_power",
        "relationship": "boss-employee"
    },
    {
        "id": "remote_work",
        "topic": "Remote Work Policy",
        "power": "high",
        "ai_position": "Office presence is important",
        "ai_opening": "I can see why remote work appeals to many employees. However, I'm concerned about team collaboration and company culture. I was wondering if we could discuss a balanced approach that addresses both needs?",
        "corpus_patterns": "high_power",
        "relationship": "boss-employee"
    }
]

ROLE_PLAY_SCENARIOS = [
    {
        "id": "friend_phone",
        "title": "Scenario 1: Disagreeing with a friend",
        "power": "low",
        "role_student": "You are talking to your friend",
        "role_ai": "Your friend",
        "situation": "Your friend thinks using a phone all day is okay. You think it's bad for health.",
        "ai_opening": "I don't think using my phone all day is bad. It's fun! I can play games and talk to my friends all the time.",
        "corpus_patterns": "low_power",
        "relationship": "friends"
    },
    {
        "id": "boss_schedule",
        "title": "Scenario 2: Negotiating with your boss",
        "power": "high",
        "role_student": "You are an employee",
        "role_ai": "Your boss",
        "situation": "Your boss says everyone must work late shifts. You have school in the morning and can't stay late.",
        "ai_opening": "We need more coverage for late shifts. Starting next week, all part-time employees will work until 11 PM. This includes you.",
        "corpus_patterns": "high_power",
        "relationship": "boss-employee"
    }
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if 'api_key' not in st.session_state:
        try:
            st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
        except (KeyError, FileNotFoundError):
            st.session_state.api_key = None
    if 'student_name' not in st.session_state:
        st.session_state.student_name = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'interaction_logs' not in st.session_state:
        st.session_state.interaction_logs = []
    if 'autonomy_log' not in st.session_state:
        st.session_state.autonomy_log = []
    if 'current_state' not in st.session_state:
        st.session_state.current_state = "welcome"
    if 'current_activity' not in st.session_state:
        st.session_state.current_activity = None
    if 'current_dialogue' not in st.session_state:
        st.session_state.current_dialogue = None
    if 'current_debate' not in st.session_state:
        st.session_state.current_debate = None
    if 'debate_turn' not in st.session_state:
        st.session_state.debate_turn = 1
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = None
    if 'temp_show_examples' not in st.session_state:
        st.session_state.temp_show_examples = False
    if 'last_audio_bytes' not in st.session_state:
        st.session_state.last_audio_bytes = None
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

def log_interaction(role: str, content: str):
    """Log an interaction"""
    st.session_state.interaction_logs.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "activity": st.session_state.current_activity,
        "state": st.session_state.current_state,
        "role": role,
        "content": content
    })

def log_autonomy(action: str):
    """Log autonomous help-seeking behavior"""
    st.session_state.autonomy_log.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "activity": st.session_state.current_activity,
        "action": action
    })

def save_logs() -> str:
    """Generate downloadable JSON log"""
    data = {
        "student_name": st.session_state.student_name,
        "session_start": st.session_state.interaction_logs[0]["timestamp"] if st.session_state.interaction_logs else None,
        "session_end": datetime.datetime.now().isoformat(),
        "interactions": st.session_state.interaction_logs,
        "autonomy_events": st.session_state.autonomy_log
    }
    return json.dumps(data, indent=2)

def call_gpt(user_message: str, relationship: str = "friend", topic: str = "") -> str:
    """Call GPT API with conversational context - WITH IMPLICIT FEEDBACK"""
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        
        # Build context based on relationship
        if relationship == "friends":
            role_context = "You are the student's friend having a casual debate. Be direct, energetic, and use casual language like 'yeah but', 'I get that, but'. Reference being friends naturally. If they're too formal, react like a friend would: 'Why so serious? We're friends!'"
        elif relationship == "classmates":
            role_context = "You are the student's classmate having a casual debate. Be friendly, direct, and use casual language like 'yeah but', 'I see what you mean but'. Reference being classmates naturally. If they're too formal, react casually: 'Relax, we're in class together!'"
        elif relationship == "boss-employee":
            role_context = "You are the student's boss in a professional discussion. Be professional, diplomatic, and use formal language like 'I understand, however', 'I can see your point, but perhaps'. Reference the professional relationship naturally. If they're too casual or direct, respond like a boss would: 'That's quite direct for a workplace conversation. I'd appreciate more diplomacy.'"
        else:
            role_context = ""
        
        # Create context message WITH FEEDBACK INSTRUCTION
        context_message = f"""{role_context}

Topic: {topic}

CRITICAL: 
1. First, evaluate if their language register matches this relationship
2. If they're TOO CASUAL with boss → give feedback like "That's quite direct..." or "I'd appreciate more diplomatic language"
3. If they're TOO FORMAL with friend → give feedback like "Why so formal? We're friends!" or "Relax, just talk naturally!"
4. Then respond authentically to their ideas and model appropriate disagreement patterns
5. Continue the debate naturally"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context_message}
        ]
        
        # Add conversation history
        messages.extend(st.session_state.conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Update conversation history
        st.session_state.conversation_history.append({"role": "user", "content": user_message})
        st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
        
        return ai_response
    
    except Exception as e:
        st.error(f"Error calling GPT: {str(e)}")
        return "I'm having trouble connecting right now. Please try again."

def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using OpenAI Whisper API"""
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        
        # Create a file-like object from bytes
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        
        # Call Whisper API
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        
        return transcript.text
    
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return ""

def show_corpus_examples(examples: List[str], title: str = "Here are some examples from real conversations:"):
    """Display corpus examples in a styled box"""
    st.markdown(f"**{title}**")
    for example in examples:
        st.markdown(f'<div class="corpus-example">"{example}"</div>', unsafe_allow_html=True)

def show_context_reminder(relationship: str, power: str):
    """Display a context reminder box"""
    if power == "low":
        if relationship == "friends":
            reminder_text = "💬 <strong>Remember:</strong> You're talking with your <strong>friend</strong> - casual and direct is fine!"
        else:
            reminder_text = "💬 <strong>Remember:</strong> You're talking with your <strong>classmate</strong> - keep it casual and friendly!"
    else:
        reminder_text = "💼 <strong>Remember:</strong> You're talking with your <strong>boss</strong> - be professional and diplomatic!"
    
    st.markdown(f'<div class="context-reminder">{reminder_text}</div>', unsafe_allow_html=True)

def display_conversation_history():
    """Display the conversation history in chat format"""
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-assistant"><strong>Partner:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

def voice_or_text_input(input_label: str, key_prefix: str, height: int = 100):
    """
    Display both voice recording and text input options for Activity 3
    Returns tuple: (input_text, input_method)
    """
    st.markdown(f"""
    <div class="voice-recording-box">
    <strong>🎤 You can either speak OR type your response:</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for voice and text input
    tab1, tab2 = st.tabs(["🎤 Voice Recording", "⌨️ Text Input"])
    
    with tab1:
        st.info("Click the microphone button below to start recording. Click again to stop.")
        
        # Audio recorder
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8453c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="3x",
        )
        
        # Check if new audio was recorded
        if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
            st.session_state.last_audio_bytes = audio_bytes
            with st.spinner("Transcribing your speech..."):
                transcribed = transcribe_audio(audio_bytes)
                st.session_state.transcribed_text = transcribed
                if transcribed:
                    st.success("✅ Recording transcribed!")
                    st.write(f"**You said:** {transcribed}")
        
        # Show previously transcribed text if exists
        elif st.session_state.transcribed_text:
            st.info(f"**Current transcription:** {st.session_state.transcribed_text}")
        
        voice_input = st.session_state.transcribed_text
    
    with tab2:
        text_input = st.text_area(input_label, height=height, key=f"{key_prefix}_text")
        voice_input = ""
    
    # Return whichever has content (prioritize voice if both exist)
    if voice_input:
        return voice_input, "voice"
    elif text_input:
        return text_input, "text"
    else:
        return "", "none"

# ============================================================================
# NOTE: The process_welcome(), process_activity1(), process_activity2(), 
# and process_activity3() functions would continue here with the rest of 
# the original implementation.
# ============================================================================
