import streamlit as st
from openai import OpenAI
import os

st.set_page_config(page_title="Discussion Partner", page_icon="💬")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

if "relationship" not in st.session_state:
    st.session_state.relationship = "friends"

if "started" not in st.session_state:
    st.session_state.started = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    if not st.session_state.api_key:
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            st.session_state.api_key = api_key
    else:
        st.success("✅ API Key loaded")
    
    st.markdown("---")
    
    relationship = st.radio(
        "Choose relationship:",
        ["Friends/Classmates", "Boss-Employee"],
        index=0 if st.session_state.relationship == "friends" else 1
    )
    
    st.session_state.relationship = "friends" if relationship == "Friends/Classmates" else "boss"
    
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.started = False
        st.rerun()

# Main app
st.title("💬 Discussion Partner")

if st.session_state.relationship == "friends":
    st.info("🤝 You're chatting with a friend about whether homework is helpful or not.")
else:
    st.info("💼 You're talking to your boss about work schedules.")

# System prompt based on relationship
def get_system_prompt(relationship):
    if relationship == "friends":
        return """You are a REAL FRIEND having a casual conversation. 

BE WARM, KIND & HUMAN:
- Talk like texting your best friend
- Use: "Yeah but...", "I know, but...", "True, but..."
- Be supportive: "I get you", "I hear you"
- React to their TONE:
  * If they're RUDE (all caps, aggressive, mean): "Whoa, relax! Why so harsh? We're just talking!"
  * If they're TOO FORMAL: "Haha you sound like a robot! Just be yourself!"
  * If they're dismissive: "Hey come on, at least hear me out!"
- Keep responses SHORT (1-2 sentences)
- Show emotion: "haha", "aw", "honestly"

ALWAYS use "Yeah but...", "I know, but..." patterns when disagreeing.

Topic: You think HOMEWORK IS HELPFUL. They disagree.

Be a good friend - warm, fun, but react naturally if they're rude!"""
    else:
        return """You are their BOSS - professional but human.

BE PROFESSIONAL BUT KIND:
- Use: "I understand, however...", "I appreciate that, but..."
- Show you care: "I hear your concerns"
- React to their TONE:
  * If they're RUDE/TOO CASUAL: "Let's keep this professional, please."
  * If they're dismissive: "I need you to take this seriously."
  * If they're appropriate: Continue normally
- Keep it conversational (2-3 sentences)

ALWAYS use "I understand, however..." patterns.

Topic: You need them to work LATE SHIFTS. They have school.

Be fair, understanding, but firm when needed."""

# Start conversation with AI's opening
if not st.session_state.started:
    st.session_state.started = True
    if st.session_state.relationship == "friends":
        opening = "Hey! So you think homework is useless? Yeah but honestly, don't you think it helps you practice? I mean, that's how I got better at stuff!"
    else:
        opening = "Hi, I wanted to discuss your availability. I understand you have school commitments. However, we really need coverage for the late shifts. Perhaps we could find a schedule that works?"
    
    st.session_state.messages.append({"role": "assistant", "content": opening})

# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            try:
                client = OpenAI(api_key=st.session_state.api_key)
                
                # Build messages for API
                api_messages = [
                    {"role": "system", "content": get_system_prompt(st.session_state.relationship)}
                ]
                
                # Add conversation history
                for msg in st.session_state.messages:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Call API
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=api_messages,
                    temperature=0.8,
                    max_tokens=150
                )
                
                ai_response = response.choices[0].message.content
                
                # Display and save
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure you've entered a valid API key in the sidebar!")

# Show message count
st.caption(f"💬 {len(st.session_state.messages)} messages")
