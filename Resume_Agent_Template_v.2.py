from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from typing import TypedDict
import os

# ==================== CONFIGURATION (EASY TO CHANGE) ====================
TARGET_COMPANY = "KPMG"                          # ← Change this to any company
TARGET_ROLE = "AI Consulting / AI Strategy"      # ← Change this to the role type
INTERNAL_REFERRAL_NAME = "Referrer’s Full Name"  # ← Put the actual name or leave as placeholder

# ==================== LOAD PERSONAL PROFILE (Gitignored) ====================
try:
    from personal_profile import user_profile
except ImportError:
    user_profile = """
    [Your real background goes here — this file is gitignored]
    Replace this with your actual resume content from personal_profile.py
    """

# ==================== LLM SETUP ====================
llm = ChatOllama(
    model="qwen2.5:14b",
    base_url="http://localhost:11434",
    temperature=0.6,
)

class AgentState(TypedDict):
    user_profile: str
    job_analysis: str
    ranked_jobs: str
    resume_bullets: str
    cover_letter: str
    final_review: str

# ==================== AGENTS ====================

def writer(state: AgentState):
    """Separate LLM calls so bullets and cover letter are distinct"""
    # Resume bullets
    bullets_prompt = f"""You are an expert resume writer for {TARGET_COMPANY}.
User profile: {state['user_profile']}

Write 6-8 strong, achievement-oriented resume bullet points ({TARGET_COMPANY} style: focus on client impact, business value, technical + consulting/communication skills).
Do not invent any achievements."""

    bullets_response = llm.invoke(bullets_prompt)

    # Cover letter
    cover_prompt = f"""You are an expert cover letter writer for {TARGET_COMPANY}.
User profile: {state['user_profile']}

Write one concise cover letter paragraph (max 7 sentences) that naturally mentions the internal referral from {INTERNAL_REFERRAL_NAME}.
Do not invent any achievements."""

    cover_response = llm.invoke(cover_prompt)

    return {
        "resume_bullets": bullets_response.content.strip(),
        "cover_letter": cover_response.content.strip()
    }

def reviewer(state: AgentState):
    """Now properly receives the content to review"""
    prompt = f"""You are a senior {TARGET_COMPANY} hiring manager reviewing application materials.

Resume Bullets:
{state.get('resume_bullets', 'No bullets provided')}

Cover Letter Paragraph:
{state.get('cover_letter', 'No cover letter provided')}

Review the above for professionalism, clarity, impact, and {TARGET_COMPANY} tone.
Fix any hallucinations or weak points and provide the final polished version."""

    response = llm.invoke(prompt)
    return {"final_review": response.content}

# ==================== BUILD THE GRAPH ====================
# (We can expand this later with Researcher, Matcher, etc.)

workflow = StateGraph(AgentState)
workflow.add_node("writer", writer)
workflow.add_node("reviewer", reviewer)

workflow.set_entry_point("writer")
workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviewer", END)

agent_crew = workflow.compile()

# ==================== RUN IT ====================
if __name__ == "__main__":
    if not user_profile.strip() or "Replace this with" in user_profile:
        print("⚠️  Warning: user_profile is empty or placeholder. Please create personal_profile.py")
    
    print(f"Starting {TARGET_COMPANY} {TARGET_ROLE} Resume Agent...\n")
    
    result = agent_crew.invoke({
        "user_profile": user_profile,
        "job_analysis": "",
        "ranked_jobs": "",
        "resume_bullets": "",
        "cover_letter": "",
        "final_review": ""
    })

    print("="*80)
    print("FINAL REVIEWED OUTPUT")
    print("="*80)
    print(result.get("final_review", "No output generated"))