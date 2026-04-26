from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from typing import TypedDict

# CodeRabbit review test - April 2026

# ==================== CONFIGURATION (EASY TO CHANGE) ====================
TARGET_COMPANY = ""                    # ← Change this to any company
TARGET_ROLE = ""   # ← Change this to the role type
INTERNAL_REFERRAL_NAME = "[Referrer's Full Name]"   # ← Put the actual name or leave as placeholder

# ==================== YOUR REAL BACKGROUND ====================
user_profile = """

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

def researcher(state: AgentState):
    prompt = f"""You are an expert recruiter for {TARGET_COMPANY}.

User background: {state['user_profile']}

Suggest 4-6 realistic {TARGET_COMPANY} roles focused on {TARGET_ROLE} based ONLY on the provided background."""

    response = llm.invoke(prompt)
    return {"job_analysis": response.content}

def matcher(state: AgentState):
    prompt = f"""You are a senior hiring manager at {TARGET_COMPANY}.

User profile: {state['user_profile']}

Internal referral available.

Rank the top 3 best-fitting roles with honest match scores (1-10) and reasoning."""

    response = llm.invoke(prompt)
    return {"ranked_jobs": response.content}

def writer(state: AgentState):
    prompt = f"""You are an expert resume and cover letter writer for {TARGET_COMPANY}.

User profile: {state['user_profile']}

Write:
1. 6-8 strong, achievement-oriented resume bullet points ({TARGET_COMPANY} style: focus on client impact, business value, technical + consulting skills)
2. One concise cover letter paragraph (max 7 sentences) that naturally mentions the internal referral.

Do not invent any achievements."""

    response = llm.invoke(prompt)
    return {"resume_bullets": response.content, "cover_letter": response.content}

def memory_compaction(state: AgentState):
    prompt = f"""Summarize the key points from all previous steps in a short, clean paragraph. Remove repetition or hallucinated content.

Previous output:
{state.get('job_analysis', '')}
{state.get('ranked_jobs', '')}
{state.get('resume_bullets', '')}
{state.get('cover_letter', '')}"""

    response = llm.invoke(prompt)
    return {"final_review": response.content}

def reviewer(state: AgentState):
    prompt = f"""You are a senior {TARGET_COMPANY} hiring manager reviewing application materials.

Review the following for professionalism, clarity, impact, and {TARGET_COMPANY} tone. Fix any hallucinated content and provide the final polished version."""

    response = llm.invoke(prompt)
    return {"final_review": response.content}

# ==================== BUILD THE GRAPH ====================

workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher)
workflow.add_node("matcher", matcher)
workflow.add_node("writer", writer)
workflow.add_node("memory_compaction", memory_compaction)
workflow.add_node("reviewer", reviewer)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "matcher")
workflow.add_edge("matcher", "writer")
workflow.add_edge("writer", "memory_compaction")
workflow.add_edge("memory_compaction", "reviewer")
workflow.add_edge("reviewer", END)

agent_crew = workflow.compile()

# ==================== RUN IT ====================

if __name__ == "__main__":
    # Validate required configuration
    if not TARGET_COMPANY or not TARGET_ROLE:
        raise ValueError(
            "Configuration Error: TARGET_COMPANY and TARGET_ROLE must be set.\n"
            "Please update lines 8-9 with your target company and role before running."
        )

    if not user_profile.strip():
        raise ValueError(
            "Configuration Error: user_profile must be filled in.\n"
            "Please update lines 13-15 with your background (experience, skills, education) before running."
        )

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
    print(f"FINAL {TARGET_COMPANY} RESUME AGENT OUTPUT")
    print("="*80)
    print(result.get("final_review", "No output generated"))