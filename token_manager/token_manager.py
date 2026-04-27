from typing import Dict, Optional, Tuple
import re
import json

class TokenManager:
    def __init__(self):
        # Common tokenizers (tiktoken works well for many models)
        self.tokenizers = {}

    def get_tokenizer(self, model_name: str):
        """Return appropriate tokenizer for the model"""
        if model_name in self.tokenizers:
            return self.tokenizers[model_name]

        # Try to load tiktoken for common families
        try:
            import tiktoken
            if any(x in model_name.lower() for x in ["gpt", "qwen", "llama", "gemma", "mistral"]):
                enc = tiktoken.get_encoding("cl100k_base")  # Good default for many models
            else:
                enc = tiktoken.get_encoding("o200k_base")   # Newer OpenAI-style
            self.tokenizers[model_name] = enc
            return enc
        except Exception:
            # Fallback: simple whitespace + punctuation count
            return None

    def count_tokens(self, text: str, model_name: str = "qwen2.5:14b") -> int:
        """Estimate number of tokens in text for a given model"""
        tokenizer = self.get_tokenizer(model_name)
        
        if tokenizer:
            return len(tokenizer.encode(text))
        else:
            # Rough fallback estimation
            return len(text.split()) + len(re.findall(r'[^\w\s]', text)) // 2 + 10

    def estimate_conversation(self, messages: list, model_name: str = "qwen2.5:14b") -> Dict:
        """Estimate tokens for a list of messages (like in a chat)"""
        total = 0
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            # Normalize content to string
            if content is None:
                content = ""
            elif isinstance(content, str):
                pass  # Already a string
            elif isinstance(content, (list, dict)):
                content = json.dumps(content, ensure_ascii=False)
            else:
                content = str(content)
            total += self.count_tokens(content, model_name)
        
        # Add overhead for system prompt, formatting, etc.
        total += 50
        
        return {
            "total_tokens": total,
            "model": model_name,
            "warning": total > 12000  # Example threshold for 14B models
        }

    def suggest_optimizations(self, text: str, max_tokens: int = 8000) -> str:
        """Simple suggestions to reduce token usage"""
        current = self.count_tokens(text)
        if current <= max_tokens:
            return f"✅ Within limit ({current}/{max_tokens} tokens)"
        
        suggestions = [
            f"Current estimate: {current} tokens (limit: {max_tokens})",
            "- Shorten or summarize long context",
            "- Remove redundant examples",
            "- Use more concise language",
            "- Split into multiple smaller calls if possible"
        ]
        return "\n".join(suggestions)

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    tm = TokenManager()
    
    sample_text = "Your long prompt or conversation history here..."
    
    print(tm.count_tokens(sample_text))
    print(tm.estimate_conversation([{"role": "user", "content": sample_text}]))
    print(tm.suggest_optimizations(sample_text, max_tokens=8000))