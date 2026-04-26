from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from typing import TypedDict
import os
import sys
import time
import logging
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION (EASY TO CHANGE) ====================
TARGET_COMPANY = "KPMG"                          # ← Change this to any company
TARGET_ROLE = "AI Consulting / AI Strategy"      # ← Change this to the role type
INTERNAL_REFERRAL_NAME = "Referrer's Full Name"  # ← Put the actual name or leave as placeholder

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds

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

# ==================== ERROR HANDLING HELPER ====================

class LLMInvocationError(Exception):
    """Raised when LLM invocation fails after all retries"""
    pass

def invoke_with_retry(llm, prompt, prompt_name="prompt", max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY_BASE):
    """
    Invoke LLM with retry logic for transient/network errors.
    
    Args:
        llm: The LLM instance to invoke
        prompt: The prompt to send to the LLM
        prompt_name: Identifier for logging (e.g., 'bullets_prompt', 'skills_prompt')
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay in seconds for exponential backoff
    
    Returns:
        LLM response object
    
    Raises:
        LLMInvocationError: If all retries fail
    """
    last_exception = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Invoking LLM for {prompt_name} (attempt {attempt}/{max_retries})")
            response = llm.invoke(prompt)
            logger.info(f"Successfully invoked LLM for {prompt_name}")
            return response
            
        except (ollama.ResponseError, TimeoutError, ConnectionError, OSError) as e:
            last_exception = e
            error_type = type(e).__name__
            logger.warning(
                f"LLM invocation failed for {prompt_name} (attempt {attempt}/{max_retries}): "
                f"{error_type} - {e!s}"
            )

            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.info(f"Retrying {prompt_name} in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed for {prompt_name}")
    
    # All retries exhausted
    raise LLMInvocationError(
        f"LLM invocation failed for {prompt_name} after {max_retries} attempts. "
        f"Last error: {type(last_exception).__name__} - {last_exception!s}"
    )

# ==================== AGENTS ====================

def writer(state: AgentState):
    """Separate LLM calls so bullets and cover letter are distinct"""
    # Resume bullets
    bullets_prompt = f"""You are an expert resume writer for {TARGET_COMPANY}.
User profile: {state['user_profile']}

Write 6-8 strong, achievement-oriented resume bullet points ({TARGET_COMPANY} style: focus on client impact, business value, technical + consulting/communication skills).
Do not invent any achievements."""

    try:
        bullets_response = invoke_with_retry(llm, bullets_prompt, prompt_name="bullets_prompt")
    except LLMInvocationError as e:
        logger.error(f"Failed to generate resume bullets: {e}")
        return {
            "resume_bullets": "[ERROR: Failed to generate resume bullets. Please check logs and retry.]",
            "cover_letter": ""
        }

    # Cover letter
    cover_prompt = f"""You are an expert cover letter writer for {TARGET_COMPANY}.
User profile: {state['user_profile']}

Write one concise cover letter paragraph (max 7 sentences) that naturally mentions the internal referral from {INTERNAL_REFERRAL_NAME}.
Do not invent any achievements."""

    try:
        cover_response = invoke_with_retry(llm, cover_prompt, prompt_name="cover_prompt")
    except LLMInvocationError as e:
        logger.error(f"Failed to generate cover letter: {e}")
        return {
            "resume_bullets": bullets_response.content.strip() if bullets_response else "",
            "cover_letter": "[ERROR: Failed to generate cover letter. Please check logs and retry.]"
        }

    return {
        "resume_bullets": bullets_response.content.strip(),
        "cover_letter": cover_response.content.strip()
    }

def reviewer(state: AgentState):
    """Now properly receives the content to review"""
    # Check for error sentinels in upstream outputs
    resume_bullets = state.get('resume_bullets', '')
    cover_letter = state.get('cover_letter', '')

    if "[ERROR:" in resume_bullets or "[ERROR:" in cover_letter:
        # Short-circuit: preserve original error messages without invoking LLM
        fallback_review = f"""RESUME BULLETS:
{resume_bullets if resume_bullets else 'No bullets provided'}

COVER LETTER:
{cover_letter if cover_letter else 'No cover letter provided'}

[ERROR: Upstream errors detected. Review skipped.]"""
        return {"final_review": fallback_review}

    prompt = f"""You are a senior {TARGET_COMPANY} hiring manager reviewing application materials.

Resume Bullets:
{resume_bullets if resume_bullets else 'No bullets provided'}

Cover Letter Paragraph:
{cover_letter if cover_letter else 'No cover letter provided'}

Review the above for professionalism, clarity, impact, and {TARGET_COMPANY} tone.
Fix any hallucinations or weak points and provide the final polished version."""

    try:
        response = invoke_with_retry(llm, prompt, prompt_name="reviewer_prompt")
    except LLMInvocationError as e:
        logger.error(f"Failed to generate final review: {e}")
        # Return fallback with original content if review fails
        fallback_review = f"""RESUME BULLETS:
{resume_bullets if resume_bullets else 'No bullets provided'}

COVER LETTER:
{cover_letter if cover_letter else 'No cover letter provided'}

[ERROR: Review generation failed. Please check logs and retry.]"""
        return {"final_review": fallback_review}

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
    if not isinstance(user_profile, str) or not user_profile.strip() or "Replace this with" in user_profile:
        logger.error("user_profile is empty or contains placeholder. Please create personal_profile.py with valid content.")
        print("⚠️  Error: user_profile is empty or placeholder. Please create personal_profile.py")
        sys.exit(1)
    
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