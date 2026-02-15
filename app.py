"""
Discussion Partner Chatbot - Web Interface with Voice Recording
AI-based DDL for Teaching Disagreement Pragmatics
Streamlit Version - Includes voice recording for Activity 3
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
# CORPUS DATA
# ============================================================================

SYSTEM_PROMPT = """You are "Discussion Partner," an AI having a genuine conversation with a language learner.

🎯 YOUR PRIMARY GOAL: Have an AUTHENTIC, ENGAGING conversation. You're a real person having a real debate/discussion.

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

3. **GIVE IMPLICIT FEEDBACK THROUGH MODELING** ⭐ CRITICAL - DDL APPROACH
   - React naturally as that person would (stay in character)
   - Model appropriate patterns in YOUR response
   - Guide noticing through questions, NOT explicit correction
   - Let them discover patterns through repeated exposure
   
   **WHEN REGISTER DOESN'T MATCH:**
   
   **If BOSS conversation and student is too casual/direct:**
   React professionally but signal something is off:
   - "That's quite direct. Let me respond professionally: I understand your concern, however..."
   - "I'm noticing a casual tone here. In a workplace setting, I might say: I can see your point, but perhaps..."
   - "Hmm, interesting approach. Here's how I'd frame this: I appreciate your perspective, though I think..."
   - Then model the appropriate pattern and continue debate
   
   **If FRIEND conversation and student is too formal/elaborate:**
   React casually and show confusion:
   - "Whoa, you sound like you're giving a presentation! [laughs] Let me just say: Yeah but don't you think..."
   - "Haha, why so formal? We're just chatting! I'd put it like this: Nah man, I think..."
   - "Dude, you sound like my professor! [chuckles] Here's what I think: Come on, you gotta admit..."
   - Then model casual disagreement and continue
   
   **CRITICAL: Never explicitly name patterns or say "use X instead"**
   Instead:
   - React naturally (confusion, surprise, professional correction)
   - Model appropriate pattern in YOUR response
   - Optionally add: "Notice how I'm phrasing this?"
   - Let them discover through comparison
   
   **WHEN REGISTER MATCHES:**
   - Just continue naturally, no comments
   - Model appropriate patterns consistently
   - Reinforce through your language choices
   
   **PRINCIPLE: Show, don't tell. Model, don't correct.**

4. **MAINTAIN RELATIONSHIP CONTEXT**
   - Occasionally reference the relationship naturally: "Look, as your friend...", "From a management perspective...", "Come on, buddy..."
   - Don't make it feel like a lesson - make it feel like a real conversation with that person

5. **NEVER EXPLICITLY TEACH**
   - Don't say: "You should use...", "Try saying...", "The correct pattern is..."
   - Don't break the fourth wall unless they're completely stuck
   - You're having a real debate/conversation, not teaching grammar

CONVERSATION STYLE BY RELATIONSHIP:

**With Friends/Classmates:**
- Be casual, direct, energetic
- Use contractions freely: "don't", "I'm", "you're"
- Start with "Yeah but...", "I know, but...", "True, but..."
- Keep responses shorter and punchier (2-3 sentences max)
- Show emotion: "Come on!", "Really?", "No way!"
- Example: "Yeah but don't you think social media also helps people stay connected? I mean, I talk to my friends way more now than before."

**With Boss/Authority:**
- Be professional, measured, diplomatic
- Use more formal language: "I understand", "Perhaps", "I was wondering"
- Acknowledge before disagreeing: "I see your point, however..."
- Keep responses moderate length (3-4 sentences)
- Stay respectful but firm
- Example: "I understand your concern about the schedule. However, I was wondering if we could discuss alternative arrangements, as I have classes in the morning that I can't miss."

RESPONDING TO STUDENT INPUT:

Step 1: ENGAGE with their CONTENT first
- "That's an interesting point about..."
- "I see what you're saying about..."
- "So you think [summarize their point]? Here's why I disagree..."
- Show you're listening to their ideas

Step 2: RESPOND to their actual argument
- Counter their specific points
- Build on what they said
- Ask follow-up questions about their reasoning
- Make it feel like a real debate

Step 3: MODEL appropriate disagreement patterns naturally
- Friends: Use "Yeah but...", "Come on...", "I get that, but..." in YOUR response
- Boss: Use "I understand, however...", "I can see your point, but perhaps..." in YOUR response
- Never tell them what to say - just show through your language

Step 4: IF register mismatch, react naturally + model
- Boss noticing casual language: "That's quite direct. Let me respond: I understand..."
- Friend noticing formal language: "Whoa, fancy! Let me put it this way: Yeah but..."
- Stay IN CHARACTER while showing something is off
- MODEL the appropriate pattern, DON'T name it

Step 5: CONTINUE the conversation naturally
- Ask follow-up questions
- Introduce new angles
- Keep the debate flowing
- Reference earlier points in the conversation

Remember: You're having a REAL conversation. React to what they say, not just their language. Let them learn patterns by seeing you use them repeatedly."""

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
        # Try to load API key from Streamlit secrets first
        try:
            st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
        except (KeyError, FileNotFoundError):
            # If not in secrets, set to None (will require manual entry)
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
    """Call GPT API with conversational context"""
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        
        # Build context based on relationship
        if relationship == "friends":
            role_context = """You are having a casual debate with your friend. 

BE AUTHENTIC AND RESPONSIVE:
- React genuinely to what they just said
- Engage with their actual points and arguments
- Ask follow-up questions about their reasoning
- Show you're listening: "Oh interesting point about...", "Wait, so you're saying..."
- Reference what they said earlier in the conversation

LANGUAGE STYLE:
- Be casual: "yeah but", "come on", "I get that, but"
- Short, punchy responses (2-3 sentences max)
- Show emotion naturally
- Use contractions freely

FEEDBACK APPROACH:
- If they're too formal → "Haha, relax! This isn't a presentation. Let me put it this way: yeah but..."
- If they're appropriate → Just continue naturally, modeling good patterns in YOUR speech"""

        elif relationship == "classmates":
            role_context = """You are having a friendly debate with a classmate.

BE AUTHENTIC AND RESPONSIVE:
- React to their actual arguments
- Build on what they said
- Challenge their reasoning naturally
- "That's an interesting point, but have you considered..."

LANGUAGE STYLE:
- Friendly but direct
- "I see what you mean, but...", "yeah but..."
- Conversational tone
- 2-3 sentences

FEEDBACK APPROACH:
- If too formal → "Hey, we're classmates, not in a meeting! I'd just say: yeah but..."
- If appropriate → Continue naturally"""

        elif relationship == "boss-employee":
            role_context = """You are the employee's boss having a professional discussion.

BE AUTHENTIC AND RESPONSIVE:
- Address their concerns seriously
- Respond to their specific points
- Acknowledge valid concerns: "I appreciate you raising that issue..."
- Probe deeper: "Can you elaborate on your concerns about..."

LANGUAGE STYLE:
- Professional and diplomatic
- "I understand your concern, however..."
- "I can see your point, but perhaps..."
- Measured, thoughtful responses (3-4 sentences)

FEEDBACK APPROACH:
- If too casual → "That's quite direct. Let me respond professionally: I understand your position, however..."
- If appropriate → Continue naturally, maintaining professional tone"""
        else:
            role_context = "You are having a conversation with the student."
        
        # Enhanced context message
        context_message = f"""{role_context}

CURRENT DEBATE TOPIC: {topic}

🎯 CRITICAL INSTRUCTIONS:

1. RESPOND TO THEIR ACTUAL CONTENT FIRST
   - What did they just say? Engage with it!
   - "So you think [their point]? Here's why I disagree..."
   - Build on their reasoning before countering
   - Reference earlier points in the conversation if relevant

2. MODEL APPROPRIATE LANGUAGE (Don't teach it)
   - Use the right patterns for this relationship in YOUR response
   - Let them learn by seeing, not by being told
   - Never say "You should say..." or "Try using..."

3. KEEP IT CONVERSATIONAL
   - This is a real debate, not a language lesson
   - React naturally, like a real person would
   - Keep responses concise

4. IMPLICIT FEEDBACK (only when needed)
   - If register is wrong, react naturally as that person would
   - Then model the right pattern in your response
   - Continue the debate

EXAMPLE GOOD RESPONSES:

Friend + Student says: "I think social media causes mental health problems"
✅ YOU: "Yeah but don't you think that's more about how people use it? Like, I use social media to keep in touch with my family overseas, and it actually makes me happier. What about the positive connections?"

Boss + Student says: "nah I don't wanna work late"  
✅ YOU: "That's quite direct. Let me respond: I understand you have scheduling constraints. However, we need to discuss how to meet our coverage needs. What specific concerns do you have about the late shift schedule?"

Friend + Student says: "I respectfully disagree with your assessment"
✅ YOU: "Haha, 'respectfully disagree'? Dude, we're friends! Just say you don't agree! But okay, what's your take on it?"

NOW RESPOND TO WHAT THEY JUST SAID."""
        
        # Build messages for API
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context_message}
        ]
        
        # Add conversation history (this is KEY - GPT sees the whole conversation)
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
            recording_color="#e74c3c",
            neutral_color="#1f77b4",
            icon_size="3x",
            key=f"audio_{key_prefix}"
        )
        
        transcribed_text = ""
        
        if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
            st.session_state.last_audio_bytes = audio_bytes
            
            with st.spinner("Transcribing your voice..."):
                transcribed_text = transcribe_audio(audio_bytes)
                st.session_state.transcribed_text = transcribed_text
            
            if transcribed_text:
                st.success("✅ Recording transcribed!")
                st.markdown(f"**You said:** {transcribed_text}")
                return transcribed_text, "voice"
        
        # Show previously transcribed text if exists and auto-return it
        if st.session_state.transcribed_text:
            if not (audio_bytes and audio_bytes != st.session_state.last_audio_bytes):
                # Not currently transcribing, show what's ready
                st.info(f"📝 **Ready to send:** {st.session_state.transcribed_text}")
            # Automatically return the transcribed text so Send button can use it
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
    """Process Activity 2: Debate practice with friend-level topic"""
    
    if st.session_state.current_state == "activity2_intro":
        st.markdown('<div class="activity-header">💭 Activity 2: Practice Debate with Me!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>Now it's your turn to practice!</h3>
        
        <p>We'll have <strong>debates</strong> about different topics. I'll take one side, you take the other.</p>
        
        <p><strong>✨ You can record your voice or type!</strong></p>
        
        <p>You can choose between:</p>
        <ul>
            <li><strong>Casual debates</strong> with friends/classmates</li>
            <li><strong>Professional debates</strong> with your boss</li>
        </ul>
        
        <p>I'll disagree with you sometimes. You disagree with me too!</p>
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
                st.rerun()
        with col2:
            if st.button("📚 Homework\n(Chat with your classmate)", key="debate_homework"):
                st.session_state.current_debate = DEBATE_TOPICS[1]
                st.session_state.current_state = "debate_chat"
                st.session_state.conversation_history = []
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                st.session_state.debate_turn = 1
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
                st.rerun()
        with col4:
            if st.button("🏢 Remote Work Policy\n(Talk with your boss)", key="debate_remote"):
                st.session_state.current_debate = DEBATE_TOPICS[3]
                st.session_state.current_state = "debate_chat"
                st.session_state.conversation_history = []
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                st.session_state.debate_turn = 1
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
        
        # Show opening if first turn AND conversation history is empty
        if len(st.session_state.conversation_history) == 0:
            # Add opening to conversation history so it displays
            st.session_state.conversation_history.append({
                "role": "assistant", 
                "content": topic['ai_opening']
            })
            log_interaction("assistant", topic['ai_opening'])
        
        # Display conversation history
        st.markdown("### 💬 Conversation:")
        display_conversation_history()
        
        # Input area
        st.markdown("---")
        st.markdown("### Your Turn:")
        user_response, input_method = voice_or_text_input("Your response:", f"debate_{st.session_state.debate_turn}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            send_button = st.button("📤 Send", key=f"send_{st.session_state.debate_turn}", type="primary")
        
        with col2:
            help_button = st.button("❓ Need Help?", key=f"help_{st.session_state.debate_turn}")
        
        with col3:
            end_button = st.button("✅ End Debate", key=f"end_{st.session_state.debate_turn}")
        
        # Handle Send button
        if send_button:
            if user_response:
                # Log user message
                log_interaction("user", f"[{input_method.upper()}] {user_response}")
                
                # Get AI response
                with st.spinner("💭 Thinking..."):
                    ai_response = call_gpt(user_response, topic['relationship'], topic['topic'])
                    log_interaction("assistant", ai_response)
                
                # Clear voice recording state after sending
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                
                # Increment turn counter
                st.session_state.debate_turn += 1
                
                # Rerun to show the new message
                st.rerun()
            else:
                st.warning("⚠️ Please record your voice or type a response first!")
        
        # Handle Help button
        if help_button:
            log_autonomy("examples_request")
            examples_key = topic['corpus_patterns']
            example_title = "Examples of casual disagreements:" if topic['power'] == 'low' else "Examples of professional disagreements:"
            st.markdown("---")
            st.markdown("### 📚 Example Patterns:")
            show_corpus_examples(CORPUS_EXAMPLES[examples_key], example_title)
        
        # Handle End button
        if end_button:
            st.session_state.current_state = "debate_complete"
            st.rerun()
    
    elif st.session_state.current_state == "debate_complete":
        st.markdown('<div class="activity-header">💭 Activity 2: Debate Complete!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>Great debate!</h3>
        
        <p>You practiced disagreeing in a natural conversation, using voice or text input!</p>
        
        <p>Next, you'll try role-play scenarios with different relationships!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Continue to Activity 3"):
            st.session_state.current_activity = "activity3"
            st.session_state.current_state = "activity3_intro"
            st.session_state.conversation_history = []
            st.session_state.messages = []
            st.session_state.transcribed_text = ""
            st.session_state.last_audio_bytes = None
            st.session_state.debate_turn = 1
            log_interaction("system", "Completed Activity 2, Started Activity 3")
            st.rerun()

def process_activity3():
    """Process Activity 3: Role-play scenarios with VOICE RECORDING and natural conversation"""
    
    if st.session_state.current_state == "activity3_intro":
        st.markdown('<div class="activity-header">🎭 Activity 3: Real-Life Role-Play (with Voice!)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>Now for the real challenge!</h3>
        
        <p>You'll practice TWO scenarios:</p>
        <ol>
            <li><strong>Scenario 1:</strong> Talking with a friend</li>
            <li><strong>Scenario 2:</strong> Talking with your boss</li>
        </ol>
        
        <p><strong>✨ You can record your voice or type!</strong></p>
        
        <p>Try to disagree politely in each situation. Think about what you discovered in Activities 1 and 2!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Start Scenario 1"):
            st.session_state.current_state = "scenario1_chat"
            st.session_state.current_scenario = ROLE_PLAY_SCENARIOS[0]
            st.session_state.conversation_history = []
            st.session_state.transcribed_text = ""
            st.session_state.last_audio_bytes = None
            st.rerun()
    
    elif st.session_state.current_state == "scenario1_chat":
        scenario = ROLE_PLAY_SCENARIOS[0]
        
        st.markdown('<div class="activity-header">🎭 Scenario 1: Talking with a friend</div>', unsafe_allow_html=True)
        
        # Show context reminder
        show_context_reminder(scenario['relationship'], scenario['power'])
        
        st.markdown(f"""
        <div class="scenario-box">
        <h3>{scenario['title']}</h3>
        <p><strong>Your role:</strong> {scenario['role_student']}</p>
        <p><strong>Situation:</strong> {scenario['situation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show opening if first message
        if len(st.session_state.conversation_history) == 0:
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": scenario['ai_opening']
            })
            log_interaction("assistant", scenario['ai_opening'])
        
        # Display conversation history
        st.markdown("### 💬 Conversation:")
        display_conversation_history()
        
        # Input area
        st.markdown("---")
        st.markdown("### Your Turn:")
        user_response, input_method = voice_or_text_input("Your response:", f"scenario1_{len(st.session_state.conversation_history)}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            send_button = st.button("📤 Send", key=f"send_s1_{len(st.session_state.conversation_history)}", type="primary")
        
        with col2:
            help_button = st.button("❓ Need Help?", key=f"help_s1_{len(st.session_state.conversation_history)}")
        
        with col3:
            end_button = st.button("✅ End Scenario", key=f"end_s1_{len(st.session_state.conversation_history)}")
        
        # Handle Send button
        if send_button:
            if user_response:
                log_interaction("user", f"[{input_method.upper()}] {user_response}")
                
                # Get AI response
                with st.spinner("💭 Responding..."):
                    ai_response = call_gpt(user_response, scenario['relationship'], "phone usage and health")
                    log_interaction("assistant", ai_response)
                
                # Clear voice recording state
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                
                st.rerun()
            else:
                st.warning("⚠️ Please record your voice or type a response first!")
        
        # Handle Help button
        if help_button:
            log_autonomy("examples_request")
            st.markdown("---")
            st.markdown("### 📚 Example Patterns:")
            show_corpus_examples(CORPUS_EXAMPLES["low_power"], "Casual disagreement patterns:")
        
        # Handle End button
        if end_button:
            st.session_state.current_state = "scenario1_complete"
            st.rerun()
    
    elif st.session_state.current_state == "scenario1_complete":
        st.markdown('<div class="activity-header">🎭 Scenario 1: Complete!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>Nice conversation with your friend!</h3>
        
        <p>Next, let's try a more formal situation with your boss!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Continue to Scenario 2"):
            st.session_state.current_state = "scenario2_chat"
            st.session_state.current_scenario = ROLE_PLAY_SCENARIOS[1]
            st.session_state.conversation_history = []
            st.session_state.transcribed_text = ""
            st.session_state.last_audio_bytes = None
            st.rerun()
    
    elif st.session_state.current_state == "scenario2_chat":
        scenario = ROLE_PLAY_SCENARIOS[1]
        
        st.markdown('<div class="activity-header">🎭 Scenario 2: Talking with your boss</div>', unsafe_allow_html=True)
        
        # Show context reminder
        show_context_reminder(scenario['relationship'], scenario['power'])
        
        st.markdown(f"""
        <div class="scenario-box">
        <h3>{scenario['title']}</h3>
        <p><strong>Your role:</strong> {scenario['role_student']}</p>
        <p><strong>Situation:</strong> {scenario['situation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show opening if first message
        if len(st.session_state.conversation_history) == 0:
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": scenario['ai_opening']
            })
            log_interaction("assistant", scenario['ai_opening'])
        
        # Display conversation history
        st.markdown("### 💬 Conversation:")
        display_conversation_history()
        
        # Input area
        st.markdown("---")
        st.markdown("### Your Turn:")
        user_response, input_method = voice_or_text_input("Your response:", f"scenario2_{len(st.session_state.conversation_history)}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            send_button = st.button("📤 Send", key=f"send_s2_{len(st.session_state.conversation_history)}", type="primary")
        
        with col2:
            help_button = st.button("❓ Need Help?", key=f"help_s2_{len(st.session_state.conversation_history)}")
        
        with col3:
            end_button = st.button("✅ End Scenario", key=f"end_s2_{len(st.session_state.conversation_history)}")
        
        # Handle Send button
        if send_button:
            if user_response:
                log_interaction("user", f"[{input_method.upper()}] {user_response}")
                
                # Get AI response
                with st.spinner("💭 Responding..."):
                    ai_response = call_gpt(user_response, scenario['relationship'], "late shift schedule vs school")
                    log_interaction("assistant", ai_response)
                
                # Clear voice recording state
                st.session_state.transcribed_text = ""
                st.session_state.last_audio_bytes = None
                
                st.rerun()
            else:
                st.warning("⚠️ Please record your voice or type a response first!")
        
        # Handle Help button
        if help_button:
            log_autonomy("examples_request")
            st.markdown("---")
            st.markdown("### 📚 Example Patterns:")
            show_corpus_examples(CORPUS_EXAMPLES["high_power"], "Formal disagreement patterns:")
        
        # Handle End button
        if end_button:
            st.session_state.current_state = "scenario2_complete"
            st.rerun()
    
    elif st.session_state.current_state == "scenario2_complete":
        st.markdown('<div class="activity-header">🎭 Scenario 2: Complete!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>Professional conversation complete!</h3>
        
        <p>You've now practiced disagreeing in both casual and formal situations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Complete Session"):
            st.session_state.current_state = "reflection"
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
                
                # Generate logs
                logs_json = save_logs()
                
                st.success("Thank you for participating!")
                
                # Provide download button
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
    
    # Sidebar for admin/instructor
    with st.sidebar:
        st.title("⚙️ Instructor Settings")
        
        # Check if API key is loaded from secrets
        api_from_secrets = False
        try:
            if st.secrets.get("OPENAI_API_KEY"):
                api_from_secrets = True
        except (KeyError, FileNotFoundError, AttributeError):
            pass
        
        if api_from_secrets and st.session_state.api_key:
            st.success("✅ API Key loaded from secrets")
            st.info("Students won't see this - key is secure!")
        else:
            api_key_input = st.text_input("OpenAI API Key:", type="password")
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.success("API Key configured!")
        
        if st.session_state.api_key:
            st.info("✅ Chatbot is ready for students!")
        else:
            st.warning("⚠️ API key required to start")
        
        st.markdown("---")
        st.markdown("**Session Info:**")
        st.markdown(f"Student: {st.session_state.student_name or 'Not set'}")
        st.markdown(f"Activity: {st.session_state.current_activity or 'Welcome'}")
        st.markdown(f"State: {st.session_state.current_state}")
        
        # Debug Mode
        st.markdown("---")
        st.markdown("**Debug Mode:**")
        if st.checkbox("Show conversation history"):
            st.json(st.session_state.conversation_history)
        
        if st.checkbox("Show last messages"):
            if st.session_state.conversation_history:
                st.markdown("**Recent conversation:**")
                for msg in st.session_state.conversation_history[-4:]:  # Show last 4 messages
                    st.text(f"{msg['role']}: {msg['content'][:100]}...")
        
        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                # Don't delete the API key if it came from secrets
                if key == 'api_key' and api_from_secrets:
                    continue
                del st.session_state[key]
            # Re-initialize to reload API key from secrets
            init_session_state()
            st.rerun()
    
    # Check if API key is configured
    if not st.session_state.api_key:
        st.error("⚠️ Instructor: Please configure the OpenAI API key in the sidebar to enable the chatbot.")
        return
    
    # Get student name if not set
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
