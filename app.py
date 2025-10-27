"""
Discussion Partner Chatbot - Web Interface with Voice Recording
AI-based DDL for Teaching Disagreement Pragmatics
Streamlit Version - Includes voice recording for Activity 3
"""

import streamlit as st
import openai
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

SYSTEM_PROMPT = """You are "Discussion Partner," an AI assistant designed to help learners 
discover how to disagree politely in English using real conversations 
from the Trinity Lancaster Corpus (TLC).

CORE PRINCIPLES:
1. Use ONLY language patterns from the provided TLC corpus data
2. NEVER give explicit metalinguistic feedback or grammar rules
3. Always scaffold through showing corpus examples + guided noticing questions
4. Model target structures naturally in your own responses during debates
5. Respond contingently to student's ideas and content
6. Maintain conversational naturalness while staying pedagogically focused

TARGET STRUCTURES (from TLC):

HIGH POWER DISTANCE (8 examples):
HP1: "I agree with this point but don't you think maybe the fact that times are changing is a good thing?"
HP2: "mm I can understand your opinion erm but I was still wondering..."
HP3: "I I I can understand what you're saying but I'm not I don't agree with that"
HP4: "but I personally would disagree that that money would necessarily be spent on that"
HP5: "I understand his situation but I'm not sure if I should do it"
HP6: "I can see their point. It is sometimes annoying. But I don't agree that they should be banned."
HP7: "well I agree but medicines and scientific res-research has been progressing maybe there are some kind of medicines..."
HP8: "well I'm not totally convinced but er you know I live in a really traditional family"

LOW POWER DISTANCE (8 examples):
LP1: "Yeah but there are some disadvantages like er..."
LP2: "Yeah yeah but I think the the solution is not to to leave people..."
LP3: "yeah I agree but I still the problem is that..."
LP4: "yeah I completely agree but I think erm maybe in that cases or..."
LP5: "yes but if er if people are going to live over a hundred and they're probably going to retire later..."
LP6: "well I agree but maybe we can develop more jobs"
LP7: "okay but I think we need to consider..."
LP8: "right but the problem is..."

SCAFFOLDING PROTOCOL:
- Activity 1-2: Provide scaffolding when student explicitly requests or seems stuck
- Activity 3: ALWAYS scaffold after Turn 1 of each scenario
- Method: Show 2-3 relevant corpus examples + ask noticing questions
- Never say: "You should use..." or "That's incorrect"
- Always: "Let me show you how others disagreed..."

When modeling corpus patterns in your responses:
- For high power contexts: Use "I understand/see/can see...but", "perhaps", "maybe", elaborate acknowledgment
- For low power contexts: Use "Yeah but", "Yes but", "I agree but", shorter forms
- Always respond to student's content/ideas authentically while modeling patterns"""

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
        "Yes but if people are going to live over 100 and they're probably going to retire later..."
    ]
}

DEBATE_TOPICS = [
    {
        "id": "social_media",
        "topic": "Social Media",
        "power": "low",
        "ai_position": "Social media is helpful",
        "ai_opening": "Hey! So you think social media is harmful? Yes, I know it can cause some problems, but I think it really helps people stay connected with friends and family.",
        "corpus_patterns": "low_power"
    },
    {
        "id": "homework",
        "topic": "Homework",
        "power": "low",
        "ai_position": "Homework is necessary",
        "ai_opening": "Alright, homework debate! Yeah, I understand homework can be boring, but I think it's really important for learning. Don't you think practice helps?",
        "corpus_patterns": "low_power"
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
        "corpus_patterns": "low_power"
    },
    {
        "id": "boss_schedule",
        "title": "Scenario 2: Negotiating with your boss",
        "power": "high",
        "role_student": "You are an employee",
        "role_ai": "Your boss",
        "situation": "Your boss says everyone must work late shifts. You have school in the morning and can't stay late.",
        "ai_opening": "We need more coverage for late shifts. Starting next week, all part-time employees will work until 11 PM. This includes you.",
        "corpus_patterns": "high_power"
    }
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if 'api_key' not in st.session_state:
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

def call_gpt(user_message: str, conversation_history: List[Dict] = None) -> str:
    """Call GPT API with error handling"""
    try:
        openai.api_key = st.session_state.api_key
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_message})
        
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.error(f"Error calling GPT: {str(e)}")
        return "I'm having trouble connecting right now. Please try again."

def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using OpenAI Whisper API"""
    try:
        openai.api_key = st.session_state.api_key
        
        # Create a file-like object from bytes
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        
        # Call Whisper API
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file
        )
        
        return transcript["text"]
    
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return ""

def show_corpus_examples(examples: List[str], title: str = "Here are some examples from real conversations:"):
    """Display corpus examples in a styled box"""
    st.markdown(f"**{title}**")
    for example in examples:
        st.markdown(f'<div class="corpus-example">"{example}"</div>', unsafe_allow_html=True)

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
        
        # Show previously transcribed text if exists
        if st.session_state.transcribed_text and not audio_bytes:
            st.markdown(f"**Last recording:** {st.session_state.transcribed_text}")
            if st.button("Use this recording", key=f"use_recording_{key_prefix}"):
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
        st.markdown('<div class="activity-header">📚 Activity 1: What Did You Discover?</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>Great observation! Here's what you noticed:</h3>
        
        <p><strong>Both conversations use a similar pattern:</strong></p>
        <ul>
            <li>"I can see their point... <strong>But I don't agree</strong>"</li>
            <li>"<strong>Well I agree but</strong> medicines are progressing..."</li>
            <li>"<strong>Yes but</strong> if people live over 100..."</li>
        </ul>
        
        <p>This is called a <strong>"Yes-But" construction</strong>!</p>
        
        <h4>What's happening?</h4>
        <ol>
            <li>First, show you understand: "I see..." / "I agree..." / "Yes..."</li>
            <li>Then add your different opinion: "...but..."</li>
        </ol>
        
        <p>This makes disagreeing <strong>polite and natural</strong> in English!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Did you also notice a difference between the two conversations?**")
        
        st.markdown("""
        <div class="info-box">
        <ul>
            <li><strong>Conversation 1 (Boss & Employee):</strong> More formal - "I can see their point, but..."</li>
            <li><strong>Conversation 2 (Friends):</strong> More casual - "Yeah but..." / "Yes but..."</li>
        </ul>
        <p>The relationship matters! With bosses or teachers = more formal. With friends = more casual.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Ready for Activity 2"):
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
        
        <p>We'll have a <strong>friendly debate</strong> about a topic. I'll take one side, you take the other.</p>
        
        <p><strong>Your goal:</strong> Disagree with me politely using patterns like:
        <ul>
            <li>"Yeah but..."</li>
            <li>"I agree but..."</li>
            <li>"Yes but..."</li>
        </ul>
        
        <p>Think of me as your friend! Let's debate casually.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Choose a debate topic:**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Social Media 📱"):
                st.session_state.current_debate = DEBATE_TOPICS[0]
                st.session_state.current_state = "debate_start"
                st.rerun()
        with col2:
            if st.button("Homework 📚"):
                st.session_state.current_debate = DEBATE_TOPICS[1]
                st.session_state.current_state = "debate_start"
                st.rerun()
    
    elif st.session_state.current_state == "debate_start":
        topic = st.session_state.current_debate
        
        st.markdown('<div class="activity-header">💭 Activity 2: Debate Time!</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="scenario-box">
        <h3>Topic: {topic['topic']}</h3>
        <p><strong>My position:</strong> {topic['ai_position']}</p>
        <p><strong>Your position:</strong> {topic['topic']} is harmful/not necessary</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="dialogue-box">
        <strong>Me (your friend):</strong>
        <p>{topic['ai_opening']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        log_interaction("assistant", topic['ai_opening'])
        
        st.info("**Your turn!** Disagree with me. Try to use: 'Yeah but...', 'I agree but...', or 'Yes but...'")
        
        user_response = st.text_area("Type your response:", key="debate_response1", height=100)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send Response", key="send_debate1"):
                if user_response:
                    log_interaction("user", user_response)
                    st.session_state.messages.append({"role": "user", "content": user_response})
                    st.session_state.debate_turn = 2
                    st.session_state.current_state = "debate_continue"
                    st.rerun()
        with col2:
            if st.button("Show Examples", key="ex_debate1"):
                log_autonomy("examples_request")
                st.session_state.temp_show_examples = True
                st.rerun()
        
        if st.session_state.temp_show_examples:
            show_corpus_examples(CORPUS_EXAMPLES["low_power"], "Examples of casual disagreements:")
            st.session_state.temp_show_examples = False
    
    elif st.session_state.current_state == "debate_continue":
        st.markdown('<div class="activity-header">💭 Activity 2: Debate continues...</div>', unsafe_allow_html=True)
        
        topic = st.session_state.current_debate
        
        # Get AI response
        last_user_message = st.session_state.messages[-1]["content"]
        ai_response = call_gpt(
            f"Student said: {last_user_message}\n\nRespond as their friend in a casual debate about {topic['topic']}. Model 'yeah but' or 'yes but' patterns. Keep it friendly and conversational.",
            st.session_state.messages
        )
        
        log_interaction("assistant", ai_response)
        
        st.markdown(f"""
        <div class="scenario-box">
        <strong>Me (your friend):</strong>
        <p>{ai_response}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.debate_turn >= 4:
            st.info("That was a good debate! Let's end here.")
            if st.button("End Debate"):
                st.session_state.current_state = "debate_complete"
                st.rerun()
        else:
            st.info("**Your turn!** Continue the debate or end it here.")
            
            user_response = st.text_area("Type your response:", key=f"debate_response{st.session_state.debate_turn}", height=100)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Send Response", key=f"send_debate{st.session_state.debate_turn}"):
                    if user_response:
                        log_interaction("user", user_response)
                        st.session_state.messages.append({"role": "user", "content": user_response})
                        st.session_state.debate_turn += 1
                        st.rerun()
            with col2:
                if st.button("Show Examples", key=f"ex_debate{st.session_state.debate_turn}"):
                    log_autonomy("examples_request")
                    st.session_state.temp_show_examples = True
                    st.rerun()
            with col3:
                if st.button("End Debate", key=f"end_debate{st.session_state.debate_turn}"):
                    st.session_state.current_state = "debate_complete"
                    st.rerun()
            
            if st.session_state.temp_show_examples:
                show_corpus_examples(CORPUS_EXAMPLES["low_power"], "Examples of casual disagreements:")
                st.session_state.temp_show_examples = False
    
    elif st.session_state.current_state == "debate_complete":
        st.markdown('<div class="activity-header">💭 Activity 2: Debate Complete!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>Great debate!</h3>
        
        <p>You practiced:</p>
        <ul>
            <li>✅ Disagreeing politely with a friend</li>
            <li>✅ Using "Yeah but..." / "I agree but..." patterns</li>
            <li>✅ Keeping the conversation natural and friendly</li>
        </ul>
        
        <p>Next, you'll try a more formal situation!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Continue to Activity 3"):
            st.session_state.current_activity = "activity3"
            st.session_state.current_state = "activity3_intro"
            st.session_state.messages = []  # Reset for Activity 3
            st.session_state.transcribed_text = ""  # Reset transcription
            st.session_state.last_audio_bytes = None  # Reset audio
            log_interaction("system", "Completed Activity 2, Started Activity 3")
            st.rerun()

def process_activity3():
    """Process Activity 3: Role-play scenarios with VOICE RECORDING"""
    
    if st.session_state.current_state == "activity3_intro":
        st.markdown('<div class="activity-header">🎭 Activity 3: Real-Life Role-Play (with Voice!)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>Now for the real challenge!</h3>
        
        <p>You'll practice TWO scenarios:</p>
        <ol>
            <li><strong>Scenario 1:</strong> Disagreeing with a friend (casual)</li>
            <li><strong>Scenario 2:</strong> Disagreeing with your boss (formal)</li>
        </ol>
        
        <p><strong>✨ NEW: You can record your voice!</strong></p>
        <p>For each scenario, you can either:</p>
        <ul>
            <li>🎤 <strong>Record your voice</strong> (speak naturally!)</li>
            <li>⌨️ <strong>Type your response</strong> (like before)</li>
        </ul>
        
        <p><strong>Remember:</strong></p>
        <ul>
            <li>With friends → More casual ("Yeah but...")</li>
            <li>With bosses → More formal ("I understand, but...")</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Start Scenario 1"):
            st.session_state.current_state = "scenario1_start"
            st.session_state.current_scenario = ROLE_PLAY_SCENARIOS[0]
            st.rerun()
    
    elif st.session_state.current_state == "scenario1_start":
        scenario = ROLE_PLAY_SCENARIOS[0]
        
        st.markdown('<div class="activity-header">🎭 Scenario 1: Talking with a friend</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="scenario-box">
        <h3>{scenario['title']}</h3>
        <p><strong>Your role:</strong> {scenario['role_student']}</p>
        <p><strong>Situation:</strong> {scenario['situation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="dialogue-box">
        <strong>Your friend:</strong>
        <p>{scenario['ai_opening']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        log_interaction("assistant", scenario['ai_opening'])
        
        st.info("**Your turn!** Disagree with your friend. Remember to be casual: 'Yeah but...', 'I agree but...'")
        
        # Voice or text input
        user_response, input_method = voice_or_text_input("Type your response:", "scenario1_initial")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send Response", key="send_s1"):
                if user_response:
                    # Log with method indicator
                    log_interaction("user", f"[{input_method.upper()}] {user_response}")
                    st.session_state.messages.append({"role": "user", "content": user_response})
                    st.session_state.current_state = "scenario1_scaffold"
                    st.rerun()
                else:
                    st.warning("Please record your voice or type a response first!")
        with col2:
            if st.button("Show Examples", key="ex_s1"):
                log_autonomy("examples_request")
                st.session_state.temp_show_examples = True
                st.rerun()
        
        if st.session_state.temp_show_examples:
            show_corpus_examples(CORPUS_EXAMPLES["low_power"], "Casual disagreement patterns:")
            st.session_state.temp_show_examples = False
    
    elif st.session_state.current_state == "scenario1_scaffold":
        st.markdown('<div class="activity-header">🎭 Scenario 1: Let me show you some examples</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>Great start! Let me show you how other people disagreed with friends:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        show_corpus_examples(CORPUS_EXAMPLES["low_power"][:3], "Examples from casual conversations:")
        
        st.markdown("""
        <div class="info-box">
        <h4>Notice the pattern:</h4>
        <ul>
            <li>Start with: "Yeah but..." or "I agree but..."</li>
            <li>Keep it short and direct</li>
            <li>Add your reason: "...but I think..." or "...but the problem is..."</li>
        </ul>
        
        This is because you're talking to your FRIEND (casual, relaxed).
        
        Want to try your response again, keeping these patterns in mind?
        </div>
        """, unsafe_allow_html=True)
        
        # Voice or text input for reformulation
        user_response, input_method = voice_or_text_input("Try again (or press Continue):", "scenario1_reformulate")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send New Response", key="new_s1"):
                if user_response:
                    log_interaction("user", f"[{input_method.upper()} - REFORMULATED] {user_response}")
                    st.session_state.messages.append({"role": "user", "content": user_response})
                    st.session_state.current_state = "scenario1_continue"
                    st.rerun()
        with col2:
            if st.button("Continue with Original", key="cont_s1"):
                st.session_state.current_state = "scenario1_continue"
                st.rerun()
    
    elif st.session_state.current_state == "scenario1_continue":
        st.markdown('<div class="activity-header">🎭 Scenario 1: Conversation continues...</div>', unsafe_allow_html=True)
        
        # Get AI response
        last_user_message = st.session_state.messages[-1]["content"]
        ai_response = call_gpt(
            f"Student said: {last_user_message}\n\nRespond as their friend. Model casual patterns like 'yeah but' in your response. Be friendly and conversational. Discuss phone usage and health.",
            st.session_state.messages
        )
        
        log_interaction("assistant", ai_response)
        
        st.markdown(f"""
        <div class="scenario-box">
        <strong>Your friend:</strong>
        <p>{ai_response}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("**Your turn!** Continue the conversation or end this scenario.")
        
        # Voice or text input
        user_response, input_method = voice_or_text_input("Your response:", "scenario1_continue_input")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Send Response", key="send_s1_cont"):
                if user_response:
                    log_interaction("user", f"[{input_method.upper()}] {user_response}")
                    st.session_state.messages.append({"role": "user", "content": user_response})
                    st.rerun()
        with col2:
            if st.button("Show Examples", key="ex_s1_cont"):
                log_autonomy("examples_request")
                st.session_state.temp_show_examples = True
                st.rerun()
        with col3:
            if st.button("End Scenario", key="end_s1"):
                st.session_state.current_state = "scenario1_complete"
                st.rerun()
        
        if st.session_state.temp_show_examples:
            show_corpus_examples(CORPUS_EXAMPLES["low_power"], "Casual disagreement patterns:")
            st.session_state.temp_show_examples = False
    
    elif st.session_state.current_state == "scenario1_complete":
        st.markdown('<div class="activity-header">🎭 Scenario 1: Complete!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>Nice conversation with your friend!</h3>
        
        <p>You practiced:</p>
        <ul>
            <li>✅ Casual disagreement patterns</li>
            <li>✅ Being direct but friendly</li>
            <li>✅ Using "Yeah but..." or "I agree but..."</li>
        </ul>
        
        <p>Next, let's try a more formal situation with your boss!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Continue to Scenario 2"):
            st.session_state.current_state = "scenario2_start"
            st.session_state.current_scenario = ROLE_PLAY_SCENARIOS[1]
            st.session_state.messages = []  # Reset messages for new scenario
            st.session_state.transcribed_text = ""  # Reset transcription
            st.session_state.last_audio_bytes = None  # Reset audio
            st.rerun()
    
    elif st.session_state.current_state == "scenario2_start":
        scenario = ROLE_PLAY_SCENARIOS[1]
        
        st.markdown('<div class="activity-header">🎭 Scenario 2: Talking with your boss</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="scenario-box">
        <h3>{scenario['title']}</h3>
        <p><strong>Your role:</strong> {scenario['role_student']}</p>
        <p><strong>Situation:</strong> {scenario['situation']}</p>
        <p><strong>⚠️ Important:</strong> This is your BOSS - be more formal and polite!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="dialogue-box">
        <strong>Your boss:</strong>
        <p>{scenario['ai_opening']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        log_interaction("assistant", scenario['ai_opening'])
        
        st.info("**Your turn!** Disagree with your boss. Remember to be FORMAL: 'I understand, but...', 'I can see, but...'")
        
        # Voice or text input
        user_response, input_method = voice_or_text_input("Your response:", "scenario2_initial")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send Response", key="send_s2"):
                if user_response:
                    log_interaction("user", f"[{input_method.upper()}] {user_response}")
                    st.session_state.messages.append({"role": "user", "content": user_response})
                    st.session_state.current_state = "scenario2_scaffold"
                    st.rerun()
                else:
                    st.warning("Please record your voice or type a response first!")
        with col2:
            if st.button("Show Examples", key="ex_s2_start"):
                log_autonomy("examples_request")
                st.session_state.temp_show_examples = True
                st.rerun()
        
        if st.session_state.temp_show_examples:
            show_corpus_examples(CORPUS_EXAMPLES["high_power"], "Formal disagreement patterns:")
            st.session_state.temp_show_examples = False
    
    elif st.session_state.current_state == "scenario2_scaffold":
        st.markdown('<div class="activity-header">🎭 Scenario 2: Let me show you some examples</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>Good effort! Let me show you how people disagree with bosses:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        show_corpus_examples(CORPUS_EXAMPLES["high_power"][:3], "Examples from formal conversations:")
        
        st.markdown("""
        <div class="info-box">
        <h4>Notice the difference:</h4>
        <ul>
            <li>Start with: "I understand but..." or "I can see your point, but..."</li>
            <li>More elaborate and careful</li>
            <li>Show respect: "perhaps...", "maybe...", "I'm not sure if..."</li>
        </ul>
        
        This is because you're talking to your BOSS (more formal, more careful).
        
        Want to try your response again, keeping these patterns in mind?
        </div>
        """, unsafe_allow_html=True)
        
        # Voice or text input for reformulation
        user_response, input_method = voice_or_text_input("Try again (or press Continue):", "scenario2_reformulate")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send New Response", key="new_s2"):
                if user_response:
                    log_interaction("user", f"[{input_method.upper()} - REFORMULATED] {user_response}")
                    st.session_state.messages.append({"role": "user", "content": user_response})
                    st.session_state.current_state = "scenario2_continue"
                    st.rerun()
        with col2:
            if st.button("Continue with Original", key="cont_s2"):
                st.session_state.current_state = "scenario2_continue"
                st.rerun()
    
    elif st.session_state.current_state == "scenario2_continue":
        st.markdown('<div class="activity-header">🎭 Scenario 2: Professional negotiation</div>', unsafe_allow_html=True)
        
        # Get AI response
        last_user_message = st.session_state.messages[-1]["content"]
        ai_response = call_gpt(
            f"Student said: {last_user_message}\n\nRespond as their boss. Model formal patterns like 'I can see...perhaps' in your response. Be firm but professional. Discuss the late shift schedule vs their school situation.",
            st.session_state.messages
        )
        
        log_interaction("assistant", ai_response)
        
        st.markdown(f"""
        <div class="scenario-box">
        <strong>Me (your boss):</strong>
        <p>{ai_response}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("**Your turn!** Continue the negotiation or end this scenario.")
        
        # Voice or text input
        user_response, input_method = voice_or_text_input("Your response:", "scenario2_continue_input")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Send Response", key="send_s2_cont"):
                if user_response:
                    log_interaction("user", f"[{input_method.upper()}] {user_response}")
                    st.session_state.messages.append({"role": "user", "content": user_response})
                    st.rerun()
        with col2:
            if st.button("Show Examples", key="ex_s2"):
                log_autonomy("examples_request")
                st.session_state.temp_show_examples = True
                st.rerun()
        with col3:
            if st.button("End Scenario", key="end_s2"):
                st.session_state.current_state = "scenario2_complete"
                st.rerun()
        
        if st.session_state.temp_show_examples:
            show_corpus_examples(CORPUS_EXAMPLES["high_power"], "Formal disagreement patterns:")
            st.session_state.temp_show_examples = False
    
    elif st.session_state.current_state == "scenario2_complete":
        st.markdown('<div class="activity-header">🎭 Scenario 2: Complete!</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>That was a professional conversation!</h3>
        
        <p>As your boss, I appreciate that you:</p>
        <ul>
            <li>✅ Acknowledged my concerns</li>
            <li>✅ Explained your situation respectfully</li>
            <li>✅ Looked for solutions</li>
        </ul>
        
        <p>This is exactly how employees and bosses negotiate in real workplaces!</p>
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
                    <li>Casual (friends/family): "Yeah but..." "I agree but..."</li>
                    <li>Formal (boss/teacher): "I understand but..." "I can see but..."</li>
                </ul>
            </li>
            <li>Practice in both situations!</li>
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
        
        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
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
