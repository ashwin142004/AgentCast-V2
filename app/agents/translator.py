from app.config import llm

def translate_script(script: str, target_language: str) -> str:
    """Translates the conversation script to the target language with code-mixing."""
    
    if target_language.lower() == "english":
        return script

    prompt = f"""
    Translate the following podcast script to {target_language}.
    
    Rules:
    1. Preserve "Host:" and "Guest:" speaker tags (keep them in English or transliterated as appropriate for the script format, but usually English tags are safer for parsing).
    2. Use "Code-Mixing": Keep technical terms (e.g., AI, CPU, Algorithm, Quantum) in English.
    3. Adapt idioms to {target_language} equivalents.
    4. Maintain the tone (Casual/Formal) of the original.
    
    Script:
    {script}
    """
    
    response = llm.invoke(prompt)
    
    # Handle list content (Gemini sometimes returns parts)
    if isinstance(response.content, list):
        return "".join([part["text"] for part in response.content if "text" in part])
    
    return response.content
