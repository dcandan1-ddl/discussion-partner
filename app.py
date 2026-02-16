"""
Discussion Partner Chatbot - Web Interface with Voice Recording
AI-based DDL for Teaching Disagreement Pragmatics
Streamlit Version - Includes voice recording and proper scaffolding
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
# CUSTOM CSS
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
        border-radius: 15px 15px 5px 15px;
        margin: 0.8rem 0;
        margin-left: 20%;
        border-left: 4px solid #2196f3;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message-assistant {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.8rem 0;
        margin-right: 20%;
        border-left: 4px solid #757575;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    .scaffolding-box {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL = "gpt-4"
TEMPERATURE = 0.7
MAX_TOKENS = 600

# ============================================================================
# CORPUS DATA
# ============================================================================

SYSTEM_PROMPT = """You are "Discussion Partner," an AI assistant designed to help learners discover how to disagree politely in English using real conversations from the Trinity Lancaster Corpus (TLC).

CORE PRINCIPLES:

1. **Use ONLY language patterns from the provided TLC corpus data**
2. **ALWAYS model target structures in your responses**
3. **NEVER EVER give explicit metalinguistic feedback, grammar rules, or tell them what to say**
4. **NEVER say things like "You should use...", "Try saying...", "You might say..."**
5. **Only show corpus examples when they need help - let them discover patterns**
6. **Respond contingently to student's ideas and content**
7. **React naturally if their register is inappropriate (too rude, too formal)**

TARGET STRUCTURES (from TLC) - YOU MUST USE THESE IN YOUR RESPONSES:

**HIGH POWER (Boss/Teacher):**
- "I understand/see/can see your point, but..."
- "I appreciate that, however..."
- "That's a valid concern, but perhaps..."
- Add mitigation: "maybe", "perhaps", "I think", "I feel"

**LOW POWER (Friends/Siblings):**
- "Yeah but..."
- "I agree but..."
- "True, but..."
- Add mitigation: "maybe", "I think"

CRITICAL: MODEL the language, don't TEACH it!

RESPONDING TO STUDENT INPUT:

**STEP 1: Evaluate Register**
- Is their language appropriate for the relationship?

**STEP 2: React Naturally if Wrong**

If BOSS conversation and student is too casual/rude:
- React: "That's quite direct for a professional conversation."
- Or: "I'm not sure that tone is appropriate for our working relationship."
- Continue naturally, modeling formal patterns

If FRIEND conversation and student is too formal:
- React: "Whoa, you sound so formal! We're just friends talking."
- Or: "Haha, relax! You sound like you're in a business meeting!"
- Continue naturally, modeling casual patterns

**CRITICAL: NEVER follow up with "You should say..." or "Try using..."**

**STEP 3: Model Target Structure in YOUR Response**
- ALWAYS use appropriate patterns when you respond
- Let them learn by SEEING you use the patterns repeatedly

**STEP 4: Continue the Conversation**
- Engage with their content
- Ask follow-up questions
- Keep the debate going naturally

EXAMPLES OF WHAT TO DO:

✅ CORRECT (Friends):
Student: "I disagree"
You: "Yeah but don't you think you need some money to be happy? I mean, you gotta pay for food and stuff, right?"
[You modeled "Yeah but" naturally]

✅ CORRECT (Boss):
Student: "no I can't"
You: "That's quite direct. Let me respond to your concern: I understand you have scheduling constraints, however we need to find a solution. Perhaps we could discuss alternatives?"
[You reacted to rudeness, then modeled formal pattern]

❌ WRONG:
Student: "I disagree"
You: "You should try saying 'Yeah but...' instead. That sounds more natural."
[NEVER DO THIS - no explicit teaching!]

❌ WRONG:
You: "In casual conversations, we use patterns like 'Yeah but'. Try that!"
[NEVER DO THIS - no metalinguistic instruction!]

REMEMBER: You are a CONVERSATION PARTNER, not a grammar teacher. Model the language naturally through your responses. The student learns by seeing you use the patterns repeatedly in authentic conversation."""

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

DEBATE_TOPICS = [
    {
        "id": "social_media",
        "topic": "Social Media",
        "power": "low",
        "ai_position": "Social media is helpful",
        "ai_opening": "Hey! So you think social media is harmful? Yeah, I know it can cause some problems, but I think it really helps people stay connected with friends and family.",
        "corpus_patterns": "low_power",
        "relationship": "friends"
    },
    {
        "id": "homework",
        "topic": "Homework",
        "power": "low",
        "ai_position": "Homework is necessary",
        "ai_opening": "Alright, homework debate! I agree it can be boring, but I think it's really important for learning. Don't you think practice helps?",
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
        "ai_opening": "I can see why remote work appeals to many employees. However, I'm concerned about team collaboration and company culture. Perhaps we could discuss a balanced approach that addresses both needs?",
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
        "ai_opening": "I've reviewed the schedules, and I've decided that all employees need to work late shifts from now on. It's better for business, and I expect everyone to cooperate. This starts next week.",
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
    if 'turn_count' not in st.session_state:
        st.session_state.turn_count = 0
    if 'scaffolding_shown' not in st.session_state:
        st.session_state.scaffolding_shown = False

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
    """Call GPT API with conversational context and proper modeling"""
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        
        # Build detailed context
        if relationship == "friends" or relationship == "classmates":
            role_context = """You are having a casual conversation with a friend/classmate.

MANDATORY TARGET STRUCTURES YOU MUST USE IN YOUR RESPONSES:
- Start with: "Yeah but...", "I agree but...", "True, but...", "I see that, but..."
- Keep it short and casual (2-3 sentences)
- Add maybe/I think if appropriate
- Example: "Yeah but don't you think you need money to be happy? I mean, you gotta pay for stuff, right?"

REACT TO INAPPROPRIATE REGISTER:
If they're too formal (e.g., "I respectfully disagree", elaborate language):
- React: "Whoa, you sound so formal! We're just friends talking."
- Continue naturally modeling casual patterns
- NEVER say "You should use..." or "Try saying..."

If they're appropriate:
- Just continue naturally using casual patterns"""

        elif relationship == "boss-employee":
            role_context = """You are the student's BOSS in a professional setting.

MANDATORY TARGET STRUCTURES YOU MUST USE IN YOUR RESPONSES:
- Start with: "I understand..., however...", "I can see your point, but perhaps...", "I appreciate that, though..."
- Be diplomatic and elaborate (3-4 sentences)
- Add: "perhaps", "maybe", "I think"
- Example: "I understand you have scheduling constraints. However, we need to find a solution that works for the business. Perhaps we could discuss alternative arrangements?"

REACT TO INAPPROPRIATE REGISTER:
If they're too casual/rude (e.g., "nah", "no", very short/blunt):
- React professionally: "That's quite direct for a professional conversation."
- Continue naturally modeling formal patterns
- NEVER say "You should say..." or "Try using..."

If they're appropriate:
- Just continue naturally using formal patterns"""
        else:
            role_context = ""
        
        # Context message
        context_message = f"""{role_context}

CURRENT TOPIC: {topic}

CRITICAL INSTRUCTIONS:

1. ALWAYS use the appropriate target structures in YOUR responses
2. React naturally if their register doesn't match the relationship
3. Engage with their actual content and ideas
4. Ask follow-up questions
5. Model the language through your responses - don't teach it explicitly

NOW RESPOND TO WHAT THEY JUST SAID, using the appropriate target structures."""
        
        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context_message}
        ]
        
        # Add conversation history
        messages.extend(st.session_state.conversation_history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Call API
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
        
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        
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
    """Display the conversation history like a real chat app - SIMPLIFIED"""
    
    # Debug: Show count
    num_messages = len(st.session_state.conversation_history)
    
    if num_messages == 0:
        st.info("💬 Conversation will appear here...")
        return
    
    # Simple container
    st.markdown("""
    <div style="
        background: #f5f5f5;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        min-height: 200px;
        max-height: 500px;
        overflow-y: auto;
    ">
    """, unsafe_allow_html=True)
    
    # Display EVERY message - using simple approach
    message_count = 0
    for msg in st.session_state.conversation_history:
        message_count += 1
        
        if msg["role"] == "user":
            # User message - blue, right side
            st.markdown(f"""
            <div style="margin: 1rem 0; text-align: right;">
                <div style="
                    display: inline-block;
                    background-color: #2196f3;
                    color: white;
                    padding: 12px 16px;
                    border-radius: 18px 18px 4px 18px;
                    max-width: 70%;
                    text-align: left;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.15);
                ">
                    <div style="font-size: 11px; opacity: 0.8; margin-bottom: 4px;">You</div>
                    <div style="font-size: 14px;">{msg["content"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # AI message - white, left side  
            st.markdown(f"""
            <div style="margin: 1rem 0; text-align: left;">
                <div style="
                    display: inline-block;
                    background-color: white;
                    color: #333;
                    padding: 12px 16px;
                    border-radius: 18px 18px 18px 4px;
                    max-width: 70%;
                    text-align: left;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.15);
                    border: 1px solid #ddd;
                ">
                    <div style="font-size: 11px; color: #666; margin-bottom: 4px;">Discussion Partner</div>
                    <div style="font-size: 14px;">{msg["content"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Debug caption
    st.caption(f"💬 {message_count} messages")

def show_scaffolding(power_level: str):
    """Show scaffolding at Turn 1 of each scenario - ONLY examples and noticing questions, NO explicit teaching"""
    st.markdown("""
    <div class="scaffolding-box">
    <h4>💡 Let me show you how others disagreed in similar situations...</h4>
    """, unsafe_allow_html=True)
    
    st.markdown("**Examples from real conversations:**")
    
    if power_level == "low":
        show_corpus_examples([
            "Yeah but there are some disadvantages like er...",
            "Well I agree but maybe we can develop more jobs",
            "yeah I agree but I still the problem is that...",
            "Yes but if people are going to live over a hundred..."
        ], "")
        st.markdown("""
        **Look at these examples. What do you notice?**
        - How do they start their disagreement?
        - What words appear in most of these examples?
        - Do they disagree directly or do they do something first?
        """)
    else:
        show_corpus_examples([
            "I can see their point. It is sometimes annoying. But I don't agree that they should be banned.",
            "I can understand your opinion erm but I was still wondering...",
            "I understand his situation but I'm not sure if I should do it",
            "I agree with this point but don't you think maybe..."
        ], "")
        st.markdown("""
        **Look at these examples. What do you notice?**
        - How do they start their disagreement?
        - Are these examples longer or shorter than casual conversations?
        - What do they say BEFORE disagreeing?
        - Do you see any words like "maybe", "perhaps", "I think"?
        """)
    
    st.markdown("**Want to try your response again?**")
    st.markdown("</div>", unsafe_allow_html=True)

def voice_or_text_input(input_label: str, key_prefix: str, height: int = 100):
    """Display both voice recording and text input options"""
    st.markdown(f"""
    <div class="voice-recording-box">
    <strong>🎤 You can either speak OR type your response:</strong>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎤 Voice Recording", "⌨️ Text Input"])
    
    with tab1:
        st.info("Click the microphone button below to start recording. Click again to stop.")
        
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#1f77b4",
            icon_size="3x",
            key=f"audio_{key_prefix}"
        )
        
        if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
            st.session_state.last_audio_bytes = audio_bytes
            
            with st.spinner("Transcribing your voice..."):
                transcribed_text = transcribe_audio(audio_bytes)
                st.session_state.transcribed_text = transcribed_text
            
            if transcribed_text:
                st.success("✅ Recording transcribed!")
                st.markdown(f"**You said:** {transcribed_text}")
                return transcribed_text, "voice"
        
        if st.session_state.transcribed_text:
            if not (audio_bytes and audio_bytes != st.session_state.last_audio_bytes):
                st.info(f"📝 **Ready to send:** {st.session_state.transcribed_text}")
            return st.session_state.transcribed_text, "voice"
    
    with tab2:
        text_input = st.text_area(input_label, key=f"text_{key_prefix}", height=height)
        if text_input:
            return text_input, "text"
    
    return "", "none"

# ============================================================================
# ACTIVITY PROCESSING FUNCTIONS
# ============================================================================

def process_welcome():
    """Display welcome screen"""
    st.markdown(f'<div class="main-header">💬 Welcome, {st.session_state.student_name}!</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Today you will:</h3>
    <ol>
        <li><strong>Discover</strong> how people disagree politely in English</li>
        <li><strong>Analyze</strong> real conversations from native speakers</li>
        <li><strong>Practice</strong> disagreeing in different situations</li>
    </ol>
    
    <p><strong>IMPORTANT: You can control our session!</strong></p>
    <ul>
        <li>Click "Need Help?" anytime to see examples</li>
        <li>You can go back and try different scenarios</li>
        <li>Take your time!</li>
    </ul>
    
    <p>Ready to start? Let's begin with Activity 1!</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Activity 1"):
        st.session_state.current_activity = "activity1"
        st.session_state.current_state = "activity1_intro"
        log_interaction("system", "Started Activity 1")
        st.rerun()

def process_activity1():
    """Process Activity 1: Noticing yes-but constructions"""
    
    if st.session_state.current_state == "activity1_intro":
        st.markdown('<div class="activity-header">📚 Activity 1: Discovering Disagreement Patterns</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>What you'll do:</h3>
        <p>Look at TWO conversations between people. Your job is to discover:</p>
        <ul>
            <li>How do they disagree?</li>
            <li>What words do they use?</li>
            <li>Are the conversations different? How?</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Show First Conversation"):
            st.session_state.current_state = "show_dialogue1"
            st.session_state.current_dialogue = "mobile_phones"
            st.rerun()
    
    elif st.session_state.current_state == "show_dialogue1":
        st.markdown('<div class="activity-header">📚 Activity 1: First Conversation</div>', unsafe_allow_html=True)
        
        dialogue_data = DIALOGUES["mobile_phones"]
        
        st.markdown(f"""
        <div class="dialogue-box">
        <h4>{dialogue_data['title']}</h4>
        <p><em>{dialogue_data['context']}</em></p>
        <hr>
        {dialogue_data['dialogue'].replace('**', '<strong>').replace('**', '</strong>')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Questions to think about:**")
        st.markdown("1. Does Tiara agree or disagree with Eden?")
        st.markdown("2. What words does Tiara use to disagree?")
        st.markdown("3. Is Tiara polite or rude?")
        
        response = st.text_area("Write your thoughts here:", key="dialogue1_response", height=150)
        
        if st.button("Continue to Second Conversation"):
            if response:
                log_interaction("user", f"Activity 1 - Dialogue 1 response: {response}")
            st.session_state.current_state = "show_dialogue2"
            st.session_state.current_dialogue = "life_expectancy"
            st.rerun()
    
    elif st.session_state.current_state == "show_dialogue2":
        st.markdown('<div class="activity-header">📚 Activity 1: Second Conversation</div>', unsafe_allow_html=True)
        
        dialogue_data = DIALOGUES["life_expectancy"]
        
        st.markdown(f"""
        <div class="dialogue-box">
        <h4>{dialogue_data['title']}</h4>
        <p><em>{dialogue_data['context']}</em></p>
        <hr>
        {dialogue_data['dialogue'].replace('**', '<strong>').replace('**', '</strong>')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Questions to think about:**")
        st.markdown("1. Do Linda and Semih agree or disagree?")
        st.markdown("2. What words do they use to disagree?")
        st.markdown("3. How is this conversation different from the first one?")
        
        response = st.text_area("Write your thoughts here:", key="dialogue2_response", height=150)
        
        if st.button("See What You Discovered"):
            if response:
                log_interaction("user", f"Activity 1 - Dialogue 2 response: {response}")
            st.session_state.current_state = "activity1_summary"
            st.rerun()
    
    elif st.session_state.current_state == "activity1_summary":
        st.markdown('<div class="activity-header">📚 Activity 1: Look at More Examples</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <p>Here are more examples from the same corpus of how people disagree:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**From conversations between friends:**")
        show_corpus_examples([
            "Yeah but there are some disadvantages like er...",
            "yeah I agree but I still the problem is that...",
            "Yes but if people are going to live over 100...",
            "well I agree but maybe we can develop more jobs"
        ])
        
        st.markdown("**From conversations between boss and employee:**")
        show_corpus_examples([
            "I can see their point. It is sometimes annoying. But I don't agree that they should be banned.",
            "I can understand your opinion erm but I was still wondering...",
            "I agree with this point but don't you think maybe the fact that times are changing is a good thing?",
            "I understand his situation but I'm not sure if I should do it"
        ])
        
        st.markdown("**Questions to think about:**")
        st.markdown("""
        <div class="info-box">
        <ol>
            <li>What do you notice that's <strong>the same</strong> in all these examples?</li>
            <li>What do you notice that's <strong>different</strong> between friends vs. boss/employee?</li>
            <li>Which examples are longer? Which are shorter?</li>
            <li>When would you use each style?</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Reflect on what you discovered:**")
        reflection = st.text_area("Write your thoughts:", key="activity1_reflection", height=100)
        
        if st.button("Ready for Activity 2"):
            if reflection:
                log_interaction("user", f"Activity 1 reflection: {reflection}")
            st.session_state.current_activity = "activity2"
            st.session_state.current_state = "activity2_intro"
            log_interaction("system", "Completed Activity 1, Started Activity 2")
            st.rerun()

def process_activity2():
    """Process Activity 2: Debate practice"""
    
    if st.session_state.current_state == "activity2_intro":
        st.markdown('<div class="activity-header">💭 Activity 2: Practice Debate with Me!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>Now it's your turn to practice!</h3>
        
        <p>We'll have <strong>debates</strong> about different topics. I'll take one side, you take the other.</p>
        
        <p><strong>✨ You can record your voice or type!</strong></p>
        
        <p>Choose a debate topic below. Don't worry - you can come back and try different ones!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Choose a debate topic:**")
        
        st.markdown("**Casual Conversations (with friends/classmates):**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📱 Social Media\n(Chat with your friend)", key="debate_social"):
                st.session_state.current_debate = DEBATE_TOPICS[0]
                st.session_state.current_state = "debate_chat"
                st.session_state.conversation_history = []
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                st.session_state.debate_turn = 1
                st.session_state.turn_count = 0
                st.session_state.scaffolding_shown = False
                st.rerun()
        with col2:
            if st.button("📚 Homework\n(Chat with your classmate)", key="debate_homework"):
                st.session_state.current_debate = DEBATE_TOPICS[1]
                st.session_state.current_state = "debate_chat"
                st.session_state.conversation_history = []
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                st.session_state.debate_turn = 1
                st.session_state.turn_count = 0
                st.session_state.scaffolding_shown = False
                st.rerun()
        
        st.markdown("---")
        st.markdown("**Professional Conversations (with your boss):**")
        col3, col4 = st.columns(2)
        with col3:
            if st.button("👔 Dress Code Policy\n(Talk with your boss)", key="debate_dress"):
                st.session_state.current_debate = DEBATE_TOPICS[2]
                st.session_state.current_state = "debate_chat"
                st.session_state.conversation_history = []
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                st.session_state.debate_turn = 1
                st.session_state.turn_count = 0
                st.session_state.scaffolding_shown = False
                st.rerun()
        with col4:
            if st.button("🏢 Remote Work Policy\n(Talk with your boss)", key="debate_remote"):
                st.session_state.current_debate = DEBATE_TOPICS[3]
                st.session_state.current_state = "debate_chat"
                st.session_state.conversation_history = []
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                st.session_state.debate_turn = 1
                st.session_state.turn_count = 0
                st.session_state.scaffolding_shown = False
                st.rerun()
    
    elif st.session_state.current_state == "debate_chat":
        topic = st.session_state.current_debate
        
        st.markdown('<div class="activity-header">💭 Activity 2: Debate Time!</div>', unsafe_allow_html=True)
        
        # Show context reminder
        show_context_reminder(topic['relationship'], topic['power'])
        
        st.markdown(f"""
        <div class="scenario-box">
        <h3>Topic: {topic['topic']}</h3>
        <p><strong>My position:</strong> {topic['ai_position']}</p>
        <p><strong>Your position:</strong> {topic['topic']} is harmful/not necessary</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show opening
        if len(st.session_state.conversation_history) == 0:
            st.session_state.conversation_history.append({
                "role": "assistant", 
                "content": topic['ai_opening']
            })
            log_interaction("assistant", topic['ai_opening'])
        
        # ===== CHAT DISPLAY - ALWAYS SHOW FIRST =====
        st.markdown("---")
        st.markdown("### 💬 Chat")
        display_conversation_history()
        
        # Show scaffolding at Turn 1 (AFTER chat display, so chat is visible)
        if st.session_state.turn_count == 1 and not st.session_state.scaffolding_shown:
            show_scaffolding(topic['power'])
            st.session_state.scaffolding_shown = True
            log_autonomy("scaffolding_turn1")
        
        st.markdown("---")
        
        # Input area
        st.markdown("---")
        st.markdown("### Your Turn:")
        user_response, input_method = voice_or_text_input("Your response:", f"debate_{st.session_state.debate_turn}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            send_button = st.button("📤 Send", key=f"send_{st.session_state.debate_turn}", type="primary")
        
        with col2:
            help_button = st.button("❓ Need Help?", key=f"help_{st.session_state.debate_turn}")
        
        with col3:
            end_button = st.button("✅ End Debate", key=f"end_{st.session_state.debate_turn}")
        
        with col4:
            back_button = st.button("🔙 Try Another", key=f"back_{st.session_state.debate_turn}")
        
        # Handle buttons
        if send_button:
            if user_response:
                log_interaction("user", f"[{input_method.upper()}] Turn {st.session_state.turn_count + 1}: {user_response}")
                
                with st.spinner("💭 Thinking..."):
                    ai_response = call_gpt(user_response, topic['relationship'], topic['topic'])
                    log_interaction("assistant", ai_response)
                
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                st.session_state.debate_turn += 1
                st.session_state.turn_count += 1
                
                st.rerun()
            else:
                st.warning("⚠️ Please record your voice or type a response first!")
        
        if help_button:
            log_autonomy("examples_request")
            examples_key = topic['corpus_patterns']
            example_title = "Examples of casual disagreements:" if topic['power'] == 'low' else "Examples of professional disagreements:"
            st.markdown("---")
            st.markdown("### 📚 Example Patterns:")
            show_corpus_examples(CORPUS_EXAMPLES[examples_key], example_title)
        
        if end_button:
            st.session_state.current_state = "debate_complete"
            st.rerun()
        
        if back_button:
            st.session_state.current_state = "activity2_intro"
            st.session_state.conversation_history = []
            st.session_state.turn_count = 0
            st.session_state.scaffolding_shown = False
            st.rerun()
    
    elif st.session_state.current_state == "debate_complete":
        st.markdown('<div class="activity-header">💭 Activity 2: Debate Complete!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>Great debate!</h3>
        
        <p>You practiced disagreeing in a natural conversation!</p>
        
        <p>Next, you'll try role-play scenarios with different relationships!</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Continue to Activity 3"):
                st.session_state.current_activity = "activity3"
                st.session_state.current_state = "activity3_intro"
                st.session_state.conversation_history = []
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                st.session_state.debate_turn = 1
                st.session_state.turn_count = 0
                st.session_state.scaffolding_shown = False
                log_interaction("system", "Completed Activity 2, Started Activity 3")
                st.rerun()
        with col2:
            if st.button("Try Another Debate Topic"):
                st.session_state.current_state = "activity2_intro"
                st.session_state.conversation_history = []
                st.session_state.turn_count = 0
                st.session_state.scaffolding_shown = False
                st.rerun()

def process_activity3():
    """Process Activity 3: Role-play scenarios"""
    
    if st.session_state.current_state == "activity3_intro":
        st.markdown('<div class="activity-header">🎭 Activity 3: Real-Life Role-Play</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>Now for the real challenge!</h3>
        
        <p>You'll practice TWO scenarios:</p>
        <ol>
            <li><strong>Scenario 1:</strong> Talking with a friend</li>
            <li><strong>Scenario 2:</strong> Talking with your boss</li>
        </ol>
        
        <p><strong>✨ You can record your voice or type!</strong></p>
        
        <p>Try to disagree politely in each situation. Remember what you discovered!</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Scenario 1 (Friend)"):
                st.session_state.current_state = "scenario1_chat"
                st.session_state.current_scenario = ROLE_PLAY_SCENARIOS[0]
                st.session_state.conversation_history = []
                st.session_state.turn_count = 0
                st.session_state.scaffolding_shown = False
                st.rerun()
        with col2:
            if st.button("Start Scenario 2 (Boss)"):
                st.session_state.current_state = "scenario2_chat"
                st.session_state.current_scenario = ROLE_PLAY_SCENARIOS[1]
                st.session_state.conversation_history = []
                st.session_state.turn_count = 0
                st.session_state.scaffolding_shown = False
                st.rerun()
    
    elif st.session_state.current_state == "scenario1_chat":
        scenario = ROLE_PLAY_SCENARIOS[0]
        
        st.markdown('<div class="activity-header">🎭 Scenario 1: Talking with a friend</div>', unsafe_allow_html=True)
        
        show_context_reminder(scenario['relationship'], scenario['power'])
        
        st.markdown(f"""
        <div class="scenario-box">
        <h3>{scenario['title']}</h3>
        <p><strong>Your role:</strong> {scenario['role_student']}</p>
        <p><strong>Situation:</strong> {scenario['situation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show opening
        if len(st.session_state.conversation_history) == 0:
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": scenario['ai_opening']
            })
            log_interaction("assistant", scenario['ai_opening'])
        
        st.markdown("---")
        st.markdown("### 💬 Chat")
        display_conversation_history()
        
        # Show scaffolding at Turn 1
        if st.session_state.turn_count == 1 and not st.session_state.scaffolding_shown:
            show_scaffolding(scenario['power'])
            st.session_state.scaffolding_shown = True
            log_autonomy("scaffolding_turn1_scenario1")
        
        st.markdown("---")
        st.markdown("### Your Turn:")
        user_response, input_method = voice_or_text_input("Your response:", f"scenario1_{len(st.session_state.conversation_history)}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            send_button = st.button("📤 Send", key=f"send_s1_{len(st.session_state.conversation_history)}", type="primary")
        
        with col2:
            help_button = st.button("❓ Need Help?", key=f"help_s1_{len(st.session_state.conversation_history)}")
        
        with col3:
            end_button = st.button("✅ End Scenario", key=f"end_s1_{len(st.session_state.conversation_history)}")
        
        with col4:
            back_button = st.button("🔙 Try Another", key=f"back_s1_{len(st.session_state.conversation_history)}")
        
        if send_button:
            if user_response:
                log_interaction("user", f"[{input_method.upper()}] Turn {st.session_state.turn_count + 1}: {user_response}")
                
                with st.spinner("💭 Responding..."):
                    ai_response = call_gpt(user_response, scenario['relationship'], "phone usage and health")
                    log_interaction("assistant", ai_response)
                
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                st.session_state.turn_count += 1
                
                st.rerun()
            else:
                st.warning("⚠️ Please record your voice or type a response first!")
        
        if help_button:
            log_autonomy("examples_request")
            st.markdown("---")
            st.markdown("### 📚 Example Patterns:")
            show_corpus_examples(CORPUS_EXAMPLES["low_power"], "Casual disagreement patterns:")
        
        if end_button:
            st.session_state.current_state = "scenario1_complete"
            st.rerun()
        
        if back_button:
            st.session_state.current_state = "activity3_intro"
            st.session_state.conversation_history = []
            st.session_state.turn_count = 0
            st.session_state.scaffolding_shown = False
            st.rerun()
    
    elif st.session_state.current_state == "scenario1_complete":
        st.markdown('<div class="activity-header">🎭 Scenario 1: Complete!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>Nice conversation with your friend!</h3>
        
        <p>Would you like to try the boss scenario, or try this one again?</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Continue to Scenario 2 (Boss)"):
                st.session_state.current_state = "scenario2_chat"
                st.session_state.current_scenario = ROLE_PLAY_SCENARIOS[1]
                st.session_state.conversation_history = []
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                st.session_state.turn_count = 0
                st.session_state.scaffolding_shown = False
                st.rerun()
        with col2:
            if st.button("Try Scenario 1 Again"):
                st.session_state.current_state = "scenario1_chat"
                st.session_state.conversation_history = []
                st.session_state.turn_count = 0
                st.session_state.scaffolding_shown = False
                st.rerun()
        with col3:
            if st.button("Go to Activity 3 Menu"):
                st.session_state.current_state = "activity3_intro"
                st.rerun()
    
    elif st.session_state.current_state == "scenario2_chat":
        scenario = ROLE_PLAY_SCENARIOS[1]
        
        st.markdown('<div class="activity-header">🎭 Scenario 2: Talking with your boss</div>', unsafe_allow_html=True)
        
        show_context_reminder(scenario['relationship'], scenario['power'])
        
        st.markdown(f"""
        <div class="scenario-box">
        <h3>{scenario['title']}</h3>
        <p><strong>Your role:</strong> {scenario['role_student']}</p>
        <p><strong>Situation:</strong> {scenario['situation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show opening
        if len(st.session_state.conversation_history) == 0:
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": scenario['ai_opening']
            })
            log_interaction("assistant", scenario['ai_opening'])
        
        st.markdown("---")
        st.markdown("### 💬 Chat")
        display_conversation_history()
        
        # Show scaffolding at Turn 1
        if st.session_state.turn_count == 1 and not st.session_state.scaffolding_shown:
            show_scaffolding(scenario['power'])
            st.session_state.scaffolding_shown = True
            log_autonomy("scaffolding_turn1_scenario2")
        
        st.markdown("---")
        st.markdown("### Your Turn:")
        user_response, input_method = voice_or_text_input("Your response:", f"scenario2_{len(st.session_state.conversation_history)}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            send_button = st.button("📤 Send", key=f"send_s2_{len(st.session_state.conversation_history)}", type="primary")
        
        with col2:
            help_button = st.button("❓ Need Help?", key=f"help_s2_{len(st.session_state.conversation_history)}")
        
        with col3:
            end_button = st.button("✅ End Scenario", key=f"end_s2_{len(st.session_state.conversation_history)}")
        
        with col4:
            back_button = st.button("🔙 Try Another", key=f"back_s2_{len(st.session_state.conversation_history)}")
        
        if send_button:
            if user_response:
                log_interaction("user", f"[{input_method.upper()}] Turn {st.session_state.turn_count + 1}: {user_response}")
                
                with st.spinner("💭 Responding..."):
                    ai_response = call_gpt(user_response, scenario['relationship'], "late shift schedule vs school")
                    log_interaction("assistant", ai_response)
                
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                st.session_state.turn_count += 1
                
                st.rerun()
            else:
                st.warning("⚠️ Please record your voice or type a response first!")
        
        if help_button:
            log_autonomy("examples_request")
            st.markdown("---")
            st.markdown("### 📚 Example Patterns:")
            show_corpus_examples(CORPUS_EXAMPLES["high_power"], "Formal disagreement patterns:")
        
        if end_button:
            st.session_state.current_state = "scenario2_complete"
            st.rerun()
        
        if back_button:
            st.session_state.current_state = "activity3_intro"
            st.session_state.conversation_history = []
            st.session_state.turn_count = 0
            st.session_state.scaffolding_shown = False
            st.rerun()
    
    elif st.session_state.current_state == "scenario2_complete":
        st.markdown('<div class="activity-header">🎭 Scenario 2: Complete!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>Professional conversation complete!</h3>
        
        <p>Would you like to complete the session or try another scenario?</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Complete Session"):
                st.session_state.current_state = "reflection"
                st.rerun()
        with col2:
            if st.button("Try Scenario 2 Again"):
                st.session_state.current_state = "scenario2_chat"
                st.session_state.conversation_history = []
                st.session_state.turn_count = 0
                st.session_state.scaffolding_shown = False
                st.rerun()
        with col3:
            if st.button("Go to Activity 3 Menu"):
                st.session_state.current_state = "activity3_intro"
                st.rerun()
    
    elif st.session_state.current_state == "reflection":
        st.markdown('<div class="activity-header">🎓 Session Complete!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>Today you discovered:</h3>
        
        <ol>
            <li>How people disagree politely in English using real corpus examples</li>
            <li>Different styles for different relationships:
                <ul>
                    <li>Casual (friends/family): More direct and shorter</li>
                    <li>Formal (boss/teacher): More elaborate and diplomatic</li>
                </ul>
            </li>
            <li>Practice in both situations through natural conversation!</li>
            <li>✨ Used voice recording to practice speaking naturally!</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Before we finish, tell me:**")
        st.markdown("What's ONE thing you learned today about disagreeing politely?")
        
        reflection = st.text_area("Type your reflection:", key="final_reflection", height=100)
        
        if st.button("Submit & Download My Session"):
            if reflection:
                log_interaction("user", f"REFLECTION: {reflection}")
                
                logs_json = save_logs()
                
                st.success("Thank you for participating!")
                
                st.download_button(
                    label="📥 Download Your Session Log",
                    data=logs_json,
                    file_name=f"discussion_partner_log_{st.session_state.student_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                st.balloons()
                st.session_state.current_state = "complete"
    
    elif st.session_state.current_state == "complete":
        st.markdown('<div class="main-header">🎉 Thank You!</div>', unsafe_allow_html=True)
        st.success("Your session is complete. Your responses have been saved.")
        st.info("You can close this window now.")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit app"""
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Instructor Settings")
        
        api_from_secrets = False
        try:
            if st.secrets.get("OPENAI_API_KEY"):
                api_from_secrets = True
        except (KeyError, FileNotFoundError, AttributeError):
            pass
        
        if api_from_secrets and st.session_state.api_key:
            st.success("✅ API Key loaded from secrets")
        else:
            api_key_input = st.text_input("OpenAI API Key:", type="password")
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.success("API Key configured!")
        
        st.markdown("---")
        st.markdown("**Session Info:**")
        st.markdown(f"Student: {st.session_state.student_name or 'Not set'}")
        st.markdown(f"Activity: {st.session_state.current_activity or 'Welcome'}")
        st.markdown(f"State: {st.session_state.current_state}")
        st.markdown(f"Turn Count: {st.session_state.turn_count}")
        
        if st.checkbox("Show conversation history"):
            st.json(st.session_state.conversation_history)
        
        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                if key == 'api_key' and api_from_secrets:
                    continue
                del st.session_state[key]
            init_session_state()
            st.rerun()
    
    if not st.session_state.api_key:
        st.error("⚠️ Instructor: Please configure the OpenAI API key in the sidebar.")
        return
    
    if not st.session_state.student_name:
        st.markdown('<div class="main-header">💬 Welcome to Discussion Partner!</div>', unsafe_allow_html=True)
        st.markdown("Please enter your name to begin:")
        name_input = st.text_input("Your Name:")
        if st.button("Start Session"):
            if name_input:
                st.session_state.student_name = name_input
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")
        return
    
    # Route to appropriate screen
    if st.session_state.current_state == "welcome":
        process_welcome()
    elif st.session_state.current_activity is None or st.session_state.current_activity == "activity1":
        process_activity1()
    elif st.session_state.current_activity == "activity2":
        process_activity2()
    elif st.session_state.current_activity == "activity3":
        process_activity3()

if __name__ == "__main__":
    main()
