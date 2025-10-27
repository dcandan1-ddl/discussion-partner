Discussion Partner Chatbot - Web Interface
AI-based DDL for Teaching Disagreement Pragmatics
Streamlit Version - No Python installation needed for students
"""

import streamlit as st
import openai
import json
import datetime
from typing import Dict, List, Optional

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

CATEGORIZATION_EXAMPLES = [
    {"id": "A", "text": "I can see their point. It is sometimes annoying. But I don't agree that they should be banned.", "category": "formal"},
    {"id": "B", "text": "Yeah but there are some disadvantages like er...", "category": "casual"},
    {"id": "C", "text": "yeah I agree but I still the problem is that...", "category": "casual"},
    {"id": "D", "text": "I can understand your opinion erm but I was still wondering...", "category": "formal"},
    {"id": "E", "text": "Well I agree but maybe we can develop more jobs", "category": "casual"},
    {"id": "F", "text": "I agree with this point but don't you think maybe the fact that times are changing is a good thing?", "category": "formal"},
    {"id": "G", "text": "Yes but if people are going to live over 100 and they're probably going to retire later...", "category": "casual"},
    {"id": "H", "text": "I understand his situation but I'm not sure if I should do it", "category": "formal"}
]

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_activity = None
        st.session_state.current_state = "welcome"
        st.session_state.conversation_log = []
        st.session_state.autonomy_log = []
        st.session_state.scaffolding_log = []
        st.session_state.categorization_index = 0
        st.session_state.scenario_turn = 0
        st.session_state.messages = []
        st.session_state.student_name = ""
        st.session_state.api_key = ""

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def log_interaction(role: str, content: str, metadata: Optional[Dict] = None):
    """Log all interactions for research purposes"""
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "student_name": st.session_state.student_name,
        "role": role,
        "content": content,
        "activity": st.session_state.current_activity,
        "state": st.session_state.current_state
    }
    if metadata:
        entry.update(metadata)
    st.session_state.conversation_log.append(entry)

def log_autonomy(request_type: str):
    """Track learner autonomy requests"""
    st.session_state.autonomy_log.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "student_name": st.session_state.student_name,
        "request": request_type,
        "activity": st.session_state.current_activity
    })

def log_scaffolding(trigger: str, examples_shown: List[str]):
    """Track when scaffolding is provided"""
    st.session_state.scaffolding_log.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "student_name": st.session_state.student_name,
        "trigger": trigger,
        "examples": examples_shown,
        "activity": st.session_state.current_activity
    })

def save_logs():
    """Generate logs JSON"""
    logs = {
        "student_name": st.session_state.student_name,
        "conversation_log": st.session_state.conversation_log,
        "autonomy_log": st.session_state.autonomy_log,
        "scaffolding_log": st.session_state.scaffolding_log,
        "session_date": datetime.datetime.now().isoformat()
    }
    return json.dumps(logs, indent=2)

# ============================================================================
# GPT FUNCTIONS
# ============================================================================

def call_gpt(user_message: str, context_messages: List[Dict] = None) -> str:
    """Call OpenAI API"""
    if not st.session_state.api_key:
        return "Error: API key not configured. Please contact your instructor."

    openai.api_key = st.session_state.api_key

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context_messages:
        messages.extend(context_messages)

    messages.append({"role": "user", "content": user_message})

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# UI DISPLAY FUNCTIONS
# ============================================================================

def show_dialogue(dialogue_key: str):
    """Display a dialogue in a nice box"""
    dialogue = DIALOGUES[dialogue_key]

    st.markdown(f"""
    <div class="dialogue-box">
        <h3>{dialogue['title']}</h3>
        <p><em>{dialogue['context']}</em></p>
        <hr>
        {dialogue['dialogue'].replace('**', '<strong>').replace('**', '</strong>')}
    </div>
    """, unsafe_allow_html=True)

def show_corpus_examples(examples: List[str], title: str = "Examples from the Corpus"):
    """Display corpus examples"""
    st.markdown(f"**{title}**")
    for i, example in enumerate(examples, 1):
        st.markdown(f"""
        <div class="corpus-example">
            {i}. {example}
        </div>
        """, unsafe_allow_html=True)

def show_scenario(text: str):
    """Display scenario text"""
    st.markdown(f"""
    <div class="scenario-box">
        {text}
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN CHATBOT LOGIC
# ============================================================================

def process_welcome():
    """Handle welcome screen"""
    st.markdown('<div class="main-header">💬 Discussion Partner</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h3>Welcome!</h3>
        <p>Hi! I'm your Discussion Partner. Today we'll explore how people disagree politely in English.</p>
        <p>I'll show you <strong>REAL conversations</strong> from the Trinity Lancaster Corpus - these are actual English speakers talking. We'll discover patterns together, then you'll practice with me!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
        <h4>⚡ You Control This Session!</h4>
        <ul>
            <li>Say "Show me that again" anytime to review</li>
            <li>Say "I need help" if you're stuck</li>
            <li>Say "Can I see more examples?" to explore further</li>
            <li>Say "Continue" when you're ready to move on</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✅ Ready to Start!"):
        st.session_state.current_state = "activity1_intro"
        st.rerun()

def process_activity1():
    """Handle Activity 1"""
    if st.session_state.current_state == "activity1_intro":
        st.markdown('<div class="activity-header">📚 Activity 1: How do people disagree politely?</div>', unsafe_allow_html=True)

        st.info("""
        We're going to look at 2 real conversations. Your job: figure out if people agree or disagree, and how you know!
        """)

        if st.button("See Dialogue 1"):
            st.session_state.current_state = "dialogue1_shown"
            st.rerun()

    elif st.session_state.current_state == "dialogue1_shown":
        st.markdown('<div class="activity-header">📚 Activity 1: Dialogue 1</div>', unsafe_allow_html=True)
        show_dialogue("mobile_phones")

        st.markdown("**Questions for you:**")
        st.markdown("1. Does Tiara agree or disagree with Eden?")
        st.markdown("2. What specific words tell you this?")

        user_response = st.text_area("Type your answer:", key="dialogue1_response", height=100)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Answer"):
                if user_response:
                    log_interaction("user", user_response)
                    st.session_state.messages.append({"role": "user", "content": user_response})
                    st.session_state.current_state = "dialogue1_feedback"
                    st.rerun()
        with col2:
            if st.button("I need help"):
                log_autonomy("help_request")
                st.session_state.current_state = "dialogue1_scaffolding"
                st.rerun()

    elif st.session_state.current_state == "dialogue1_scaffolding":
        st.markdown('<div class="activity-header">📚 Activity 1: Let me help you!</div>', unsafe_allow_html=True)
        show_dialogue("mobile_phones")

        st.markdown("""
        Let me show you the dialogue again. This time, look carefully at the EXACT words Tiara uses...
        
        Look at these parts:
        - **"I can see their point"**
        - **"But I don't agree"**
        
        What does Tiara do BEFORE saying "I don't agree"?
        """)

        if st.button("I understand now - Continue"):
            st.session_state.current_state = "dialogue1_transition"
            st.rerun()

    elif st.session_state.current_state == "dialogue1_feedback":
        st.markdown('<div class="activity-header">📚 Activity 1: Great thinking!</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box">
        Yes, Tiara disagrees! She does TWO things:
        <ol>
            <li>First, she acknowledges: "I can see their point"</li>
            <li>Then she disagrees: "But I don't agree..."</li>
        </ol>
        
        This is interesting! She doesn't just say "No" or "I disagree." She shows she understands FIRST.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Continue to Dialogue 2"):
            st.session_state.current_state = "dialogue2_shown"
            st.rerun()

    elif st.session_state.current_state == "dialogue1_transition":
        st.success("Great! Let's move to Dialogue 2.")
        if st.button("See Dialogue 2"):
            st.session_state.current_state = "dialogue2_shown"
            st.rerun()

    elif st.session_state.current_state == "dialogue2_shown":
        st.markdown('<div class="activity-header">📚 Activity 1: Dialogue 2</div>', unsafe_allow_html=True)

        st.info("Now let's look at a conversation between FRIENDS. Think about: Will it be the same or different?")

        show_dialogue("life_expectancy")

        st.markdown("**Questions:**")
        st.markdown("1. Do Linda and Semih agree or disagree with each other?")
        st.markdown("2. How is this SIMILAR to Dialogue 1 (Eden and Tiara)?")
        st.markdown("3. How is this DIFFERENT from Dialogue 1?")

        user_response = st.text_area("Type your observations:", key="dialogue2_response", height=150)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Answer", key="submit_d2"):
                if user_response:
                    log_interaction("user", user_response)
                    st.session_state.messages.append({"role": "user", "content": user_response})
                    st.session_state.current_state = "dialogue2_feedback"
                    st.rerun()
        with col2:
            if st.button("I need help", key="help_d2"):
                log_autonomy("help_request")
                st.session_state.current_state = "dialogue2_scaffolding"
                st.rerun()

    elif st.session_state.current_state == "dialogue2_scaffolding":
        st.markdown('<div class="activity-header">📚 Activity 1: Comparing the dialogues</div>', unsafe_allow_html=True)

        st.markdown("""
        Let me show you both dialogues side by side:
        
        **DIALOGUE 1 (Boss → Employee):**
        "I can see their point. It is sometimes annoying. But I don't agree that they should be banned."
        
        **DIALOGUE 2 (Friend → Friend):**
        "Well I agree but maybe we can develop more jobs..."
        "Yes but if people live over 100..."
        
        Look at how they START their disagreement. What words do they use?
        
        - In Dialogue 1: "I can see..." and "But I don't agree..."
        - In Dialogue 2: "Well I agree but..." and "Yes but..."
        
        Do you see anything shorter or more casual in Dialogue 2?
        """)

        if st.button("I see the pattern - Continue"):
            st.session_state.current_state = "dialogue2_feedback"
            st.rerun()

    elif st.session_state.current_state == "dialogue2_feedback":
        st.markdown('<div class="activity-header">📚 Activity 1: Excellent observation!</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box">
        <h4>You've discovered the pattern!</h4>
        
        <strong>Dialogue 1 (Boss/Employee): Longer, more careful</strong>
        <br>→ "I can see their point. It is sometimes annoying. But I don't agree..."
        
        <br><br><strong>Dialogue 2 (Friends): Shorter, more casual</strong>
        <br>→ "Yes but..."
        <br>→ "Well I agree but..."
        
        <p>People disagree differently depending on WHO they're talking to!</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Continue to Activity 2"):
            st.session_state.current_activity = "activity2"
            st.session_state.current_state = "activity2_intro"
            st.session_state.categorization_index = 0
            st.rerun()

def process_activity2():
    """Handle Activity 2"""
    if st.session_state.current_state == "activity2_intro":
        st.markdown('<div class="activity-header">🏷️ Activity 2: When do you use each style?</div>', unsafe_allow_html=True)

        st.info("""
        You noticed that people disagree differently depending on the situation!
        
        Now I'm going to show you 8 REAL examples from the Trinity Lancaster Corpus.
        
        **Your task:** Decide if each example is better for:
        - Talking to FRIEND/FAMILY (more casual)
        - Talking to TEACHER/BOSS (more formal)
        
        I'll show you ONE example at a time.
        """)

        if st.button("Start Categorizing"):
            st.session_state.current_state = "categorization"
            st.rerun()

    elif st.session_state.current_state == "categorization":
        if st.session_state.categorization_index >= len(CATEGORIZATION_EXAMPLES):
            st.session_state.current_state = "categorization_summary"
            st.rerun()
            return

        example = CATEGORIZATION_EXAMPLES[st.session_state.categorization_index]

        st.markdown('<div class="activity-header">🏷️ Activity 2: Categorize This Example</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="corpus-example">
        <h4>Example {example['id']}:</h4>
        "{example['text']}"
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Is this better for:**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("1️⃣ Friend/Family", key=f"cat_1_{example['id']}"):
                log_interaction("user", f"Example {example['id']}: Friend/Family")
                st.session_state.categorization_index += 1
                st.rerun()
        with col2:
            if st.button("2️⃣ Teacher/Boss", key=f"cat_2_{example['id']}"):
                log_interaction("user", f"Example {example['id']}: Teacher/Boss")
                st.session_state.categorization_index += 1
                st.rerun()

        st.markdown(f"**Progress:** {st.session_state.categorization_index + 1}/8 examples")

    elif st.session_state.current_state == "categorization_summary":
        st.markdown('<div class="activity-header">🏷️ Activity 2: Summary</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box">
        <h3>Great work! You've categorized all 8 examples.</h3>
        
        <h4>FRIEND/FAMILY (More casual):</h4>
        <ul>
            <li>"Yeah but there are some disadvantages like er..."</li>
            <li>"yeah I agree but I still the problem is that..."</li>
            <li>"Well I agree but maybe we can develop more jobs"</li>
            <li>"Yes but if people are going to live over 100..."</li>
        </ul>
        
        <h4>TEACHER/BOSS (More formal):</h4>
        <ul>
            <li>"I can see their point. It is sometimes annoying. But I don't agree..."</li>
            <li>"I can understand your opinion erm but I was still wondering..."</li>
            <li>"I agree with this point but don't you think maybe..."</li>
            <li>"I understand his situation but I'm not sure if I should do it"</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.info("Do you see the patterns? Casual uses 'Yeah but' / 'Yes but', while formal uses 'I understand/see...but'")

        if st.button("Continue to Activity 3 - Practice Time!"):
            st.session_state.current_activity = "activity3"
            st.session_state.current_state = "activity3_intro"
            st.rerun()

def process_activity3():
    """Handle Activity 3 - Role Play"""
    if st.session_state.current_state == "activity3_intro":
        st.markdown('<div class="activity-header">🎭 Activity 3: Practice Time - Let\'s Debate!</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        Now YOU'LL practice disagreeing with me in two situations:
        
        <strong>Scenario 1:</strong> We're siblings arguing about money (casual)
        <br><strong>Scenario 2:</strong> I'm your boss, you're my employee (formal)
        
        <h4>⚠️ IMPORTANT:</h4>
        <ul>
            <li>I'll disagree with you too (it's practice!)</li>
            <li>There's no "right answer" - we're just having conversations</li>
            <li>I'll show you examples from the corpus to help you</li>
            <li>You can ask to see examples anytime!</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Start Scenario 1"):
            st.session_state.current_state = "scenario1_start"
            st.session_state.scenario_turn = 0
            st.rerun()

    elif st.session_state.current_state == "scenario1_start":
        st.markdown('<div class="activity-header">🎭 Scenario 1: Siblings arguing about money</div>', unsafe_allow_html=True)

        show_scenario("""
        **The Setup:**
        We're siblings. I believe money is the most important thing in life.
        You disagree - you think other things are more important.
        
        Let's debate! I'll start:
        """)

        st.markdown("""
        <div class="scenario-box">
        <strong>Me (your sibling):</strong>
        <p>"I think money is the MOST important thing in life! You can't be happy without money. If you have money, you can do anything you want, buy anything, go anywhere. Don't you agree?"</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("**Your turn!** Disagree with me. Remember: we're siblings, so be natural!")

        user_response = st.text_area("Type your response:", key="scenario1_turn1", height=100)

        if st.button("Send Response"):
            if user_response:
                log_interaction("user", user_response)
                st.session_state.messages.append({"role": "user", "content": user_response})
                st.session_state.scenario_turn = 1
                st.session_state.current_state = "scenario1_scaffolding"
                st.rerun()

    elif st.session_state.current_state == "scenario1_scaffolding":
        st.markdown('<div class="activity-header">🎭 Scenario 1: Let me show you some examples!</div>', unsafe_allow_html=True)

        st.info("Before I respond, let me show you how other people disagreed with their friends and siblings in similar debates.")

        examples = CORPUS_EXAMPLES["low_power"][:3]
        log_scaffolding("turn1_automatic", examples)
        show_corpus_examples(examples)

        st.markdown("""
        **Notice how they start their disagreement?**
        - They say "Yeah but..." or "I agree but..."
        - They acknowledge first, THEN disagree
        
        Since we're siblings (casual relationship), you might try:
        - Starting with "Yeah but..."
        - Or "I agree money is important but..."
        
        Want to try your response again, using one of these patterns?
        """)

        user_response = st.text_area("Try again (or press Continue to keep your original response):", key="scenario1_reformulate", height=100)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send New Response"):
                if user_response:
                    log_interaction("user", user_response + " [REFORMULATED]")
                    st.session_state.messages.append({"role": "user", "content": user_response})
                    st.session_state.current_state = "scenario1_continue"
                    st.rerun()
        with col2:
            if st.button("Continue with Original"):
                st.session_state.current_state = "scenario1_continue"
                st.rerun()

    elif st.session_state.current_state == "scenario1_continue":
        st.markdown('<div class="activity-header">🎭 Scenario 1: Continuing the debate</div>', unsafe_allow_html=True)

        # Get AI response
        last_user_message = st.session_state.messages[-1]["content"]
        ai_response = call_gpt(
            f"Student said: {last_user_message}\n\nContinue the sibling debate naturally. Model corpus patterns like 'I agree...but maybe' in your response. Keep it conversational as siblings. Respond to their content about money vs family/other values.",
            st.session_state.messages
        )

        log_interaction("assistant", ai_response)

        st.markdown(f"""
        <div class="scenario-box">
        <strong>Me (your sibling):</strong>
        <p>{ai_response}</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("**Your turn!** Continue the debate or end this scenario.")

        user_response = st.text_area("Type your response:", key="scenario1_continue_input", height=100)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Send Response", key="send_s1"):
                if user_response:
                    log_interaction("user", user_response)
                    st.session_state.messages.append({"role": "user", "content": user_response})
                    st.rerun()
        with col2:
            if st.button("Show Examples"):
                log_autonomy("examples_request")
                st.session_state.temp_show_examples = True
                st.rerun()
        with col3:
            if st.button("End Scenario"):
                st.session_state.current_state = "scenario1_complete"
                st.rerun()

        if hasattr(st.session_state, 'temp_show_examples') and st.session_state.temp_show_examples:
            show_corpus_examples(CORPUS_EXAMPLES["low_power"], "Casual disagreement patterns:")
            st.session_state.temp_show_examples = False

    elif st.session_state.current_state == "scenario1_complete":
        st.markdown('<div class="activity-header">🎭 Scenario 1: Complete!</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box">
        <h3>Great debate!</h3>
        <p>You made strong arguments! I still think money matters, but you definitely made me think!</p>
        
        <p>Ready for Scenario 2? This time I'll be your <strong>BOSS</strong>, so think about how that might change how you disagree with me!</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Scenario 2"):
                st.session_state.current_state = "scenario2_start"
                st.session_state.scenario_turn = 0
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Review Corpus Examples"):
                log_autonomy("review_request")
                st.session_state.current_state = "scenario1_review"
                st.rerun()

    elif st.session_state.current_state == "scenario1_review":
        st.markdown('<div class="activity-header">📚 Review: Casual Disagreement Patterns</div>', unsafe_allow_html=True)
        show_corpus_examples(CORPUS_EXAMPLES["low_power"], "Casual patterns (friends/siblings):")

        if st.button("Ready for Scenario 2"):
            st.session_state.current_state = "scenario2_start"
            st.session_state.scenario_turn = 0
            st.session_state.messages = []
            st.rerun()

    elif st.session_state.current_state == "scenario2_start":
        st.markdown('<div class="activity-header">🎭 Scenario 2: Boss and Employee</div>', unsafe_allow_html=True)

        show_scenario("""
        **The Setup:**
        Now I'm your BOSS at a company. You're my employee.
        I'm going to tell you about a new policy you don't agree with.
        
        **Think:** How will you disagree with your BOSS differently than with your sibling?
        """)

        st.markdown("""
        <div class="scenario-box">
        <strong>Me (your boss):</strong>
        <p>"I've reviewed the schedules, and I've decided that all employees need to work late shifts from now on. It's better for business, and I expect everyone to cooperate. This starts next week."</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("**The situation:** You have school in the morning, so you CAN'T work late shifts. How will you disagree with me? Remember: I'm your boss!")

        user_response = st.text_area("Type your response:", key="scenario2_turn1", height=100)

        if st.button("Send Response", key="send_s2_t1"):
            if user_response:
                log_interaction("user", user_response)
                st.session_state.messages.append({"role": "user", "content": user_response})
                st.session_state.scenario_turn = 1
                st.session_state.current_state = "scenario2_scaffolding"
                st.rerun()

    elif st.session_state.current_state == "scenario2_scaffolding":
        st.markdown('<div class="activity-header">🎭 Scenario 2: Professional disagreement examples</div>', unsafe_allow_html=True)

        st.info("Before I respond, let me show you how employees disagreed with their bosses in similar situations in the corpus.")

        examples = CORPUS_EXAMPLES["high_power"][:4]
        log_scaffolding("turn1_automatic", examples)
        show_corpus_examples(examples, "Examples from Boss/Employee conversations:")

        st.markdown("""
        **Notice the differences from sibling conversations?**
        - More elaborate: "I can understand..." "I can see..."
        - Acknowledge boss's point FIRST
        - Use softer language: "I'm not sure" "maybe" "I was wondering"
        
        This is because you're talking to your BOSS (more formal, more careful).
        
        Want to try your response again, keeping these patterns in mind?
        """)

        user_response = st.text_area("Try again (or press Continue to keep your original response):", key="scenario2_reformulate", height=100)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send New Response", key="new_s2"):
                if user_response:
                    log_interaction("user", user_response + " [REFORMULATED]")
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

        user_response = st.text_area("Type your response:", key="scenario2_continue_input", height=100)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Send Response", key="send_s2_cont"):
                if user_response:
                    log_interaction("user", user_response)
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

        if hasattr(st.session_state, 'temp_show_examples') and st.session_state.temp_show_examples:
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
