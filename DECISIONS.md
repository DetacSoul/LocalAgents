# DECISIONS LOG

> After each working session with any LLM, add an entry below.

---

## Template

```
### YYYY-MM-DD — [Brief Topic]
**Model:** Claude / Grok / Gemini  
**Task:** What you asked for  
**Outcome:** What was produced or decided  
**Files changed:** List of files  
**Open items:** What's unresolved  
```

---

## Sessions

### 2026-04-26 — Resume agent v1 and v2
**Model:** Grok  
**Task:** Build a resume/cover letter agent with LangGraph + Ollama  
**Outcome:** Working 2-node pipeline (writer → reviewer) with retry logic  
**Files changed:** Resume_Agent_Template_v.1.py, Resume_Agent_Template_v.2.py  
**Open items:** Needs config externalization, README, folder structure  

### 2026-04-26 — Monorepo template and coordination setup
**Model:** Claude  
**Task:** Create reusable project template, plan multi-LLM coordination workflow  
**Outcome:** Monorepo structure with shared utils, PROJECT_SPEC, DECISIONS log, per-agent READMEs  
**Files changed:** Full template created  
**Open items:** Migrate existing flat files into subdirectories  
