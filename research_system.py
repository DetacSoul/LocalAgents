from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from typing import TypedDict
import time
import random
import logging
import ollama

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

llm = ChatOllama(
    model="qwen2.5:14b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

class ResearchState(TypedDict):
    topic: str
    trend_research: str
    opportunity_analysis: str
    risk_evaluation: str
    compounding_assessment: str
    final_recommendation: str

# ==================== HELPER FUNCTIONS ====================

def invoke_with_retry(fn, *args, max_retries=3, **kwargs):
    """Retry wrapper for LLM invoke calls with exponential backoff and jitter."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except (ollama.ResponseError, TimeoutError, ConnectionError, OSError) as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                raise

            wait_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)

# ==================== AGENTS ====================

def trend_researcher(state: ResearchState):
    prompt = f"""You are an expert trend researcher.
Topic: {state['topic']}

Provide a concise but thorough overview of the current state of this topic, including recent developments, key players, and momentum."""
    wrapped_response = invoke_with_retry(llm.invoke, prompt)
    return {"trend_research": wrapped_response.content}

def opportunity_analyzer(state: ResearchState):
    research_notes = state.get("trend_research", "")
    prompt = f"""You are an expert opportunity analyst.
Topic: {state['topic']}

Previous Research:
{research_notes}

Based on the research, identify the biggest potential opportunities, applications, and leverage points."""
    wrapped_response = invoke_with_retry(llm.invoke, prompt)
    return {"opportunity_analysis": wrapped_response.content}

def risk_evaluator(state: ResearchState):
    research_notes = state.get("trend_research", "")
    opportunity_notes = state.get("opportunity_analysis", "")
    prompt = f"""You are a critical risk and feasibility evaluator.
Topic: {state['topic']}

Previous Research:
{research_notes}

Identified Opportunities:
{opportunity_notes}

Be honest and thorough. What are the major risks, challenges, competition, technical hurdles, time/cost realities, and reasons this might not work?"""
    wrapped_response = invoke_with_retry(llm.invoke, prompt)
    return {"risk_evaluation": wrapped_response.content}

def compounding_assessor(state: ResearchState):
    research_notes = state.get("trend_research", "")
    opportunity_notes = state.get("opportunity_analysis", "")
    risk_notes = state.get("risk_evaluation", "")
    prompt = f"""You are an expert at evaluating compounding potential.
Topic: {state['topic']}

Previous Research:
{research_notes}

Identified Opportunities:
{opportunity_notes}

Risk Assessment:
{risk_notes}

How much long-term optionality, network effects, skill stacking, or exponential growth potential does this direction have?"""
    wrapped_response = invoke_with_retry(llm.invoke, prompt)
    return {"compounding_assessment": wrapped_response.content}

def final_recommender(state: ResearchState):
    opportunity_notes = state.get("opportunity_analysis", "")
    risk_notes = state.get("risk_evaluation", "")
    compounding_notes = state.get("compounding_assessment", "")
    prompt = f"""You are a final strategic recommender.

Topic: {state['topic']}

Opportunity Analysis:
{opportunity_notes}

Risk Evaluation:
{risk_notes}

Compounding Assessment:
{compounding_notes}

Synthesize all previous analysis and give a clear recommendation:
- Should this be pursued seriously?
- What is the realistic upside vs downside?
- What should be the next 1-3 concrete actions?"""

    wrapped_response = invoke_with_retry(llm.invoke, prompt)
    return {"final_recommendation": wrapped_response.content}

# ==================== BUILD THE GRAPH ====================

workflow = StateGraph(ResearchState)

workflow.add_node("trend_researcher", trend_researcher)
workflow.add_node("opportunity_analyzer", opportunity_analyzer)
workflow.add_node("risk_evaluator", risk_evaluator)
workflow.add_node("compounding_assessor", compounding_assessor)
workflow.add_node("final_recommender", final_recommender)

workflow.set_entry_point("trend_researcher")
workflow.add_edge("trend_researcher", "opportunity_analyzer")
workflow.add_edge("opportunity_analyzer", "risk_evaluator")
workflow.add_edge("risk_evaluator", "compounding_assessor")
workflow.add_edge("compounding_assessor", "final_recommender")
workflow.add_edge("final_recommender", END)

research_system = workflow.compile()

# ==================== RUN IT ====================

if __name__ == "__main__":
    topic = input("Enter the topic or idea you want to research: ")

    if not topic or not topic.strip():
        print("Please enter a non-empty topic.")
        exit()

    print(f"\nStarting Research System for: {topic}\n")

    result = research_system.invoke({
        "topic": topic,
        "trend_research": "",
        "opportunity_analysis": "",
        "risk_evaluation": "",
        "compounding_assessment": "",
        "final_recommendation": ""
    })

    print("="*80)
    print("FINAL RESEARCH SYSTEM OUTPUT")
    print("="*80)
    print(result["final_recommendation"])