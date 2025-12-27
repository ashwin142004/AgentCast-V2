from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import PodcastRequest, PodcastResponse, TranslationRequest, TranslationResponse
from app.workflow import build_graph

app = FastAPI(title="AgentCast V2 API")
graph = build_graph()

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"❌ Server Error: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.get("/")
def health_check():
    return {"status": "ok", "service": "AgentCast V2"}

@app.post("/generate-podcast", response_model=PodcastResponse)
async def generate_podcast(request: PodcastRequest):
    print(f"🚀 Receiving Request: {request}")
    
    initial_state = {
        "topic": request.topic,
        "messages": [],
        "dcs_analysis": {},
        "turn_count": 0,
        "target_language": "English",
        "final_script": ""
    }
    
    # Run graph to completion
    result = graph.invoke(initial_state)
    
    return PodcastResponse(
        status="completed",
        script=result["final_script"],
        original_script=None, # Could capture intermediate if needed
        language="English"
    )

@app.post("/translate", response_model=TranslationResponse, response_model_exclude_none=True)
async def translate_text_endpoint(request: TranslationRequest):
    print(f"Loading Translator for: {request.target_language}")
    
    # Ensure this import path matches your folder structure
    from app.agents.translator import translate_script, translate_dialogue
    
    if request.script:
        translated_script = translate_dialogue(request.script, request.target_language)
        return TranslationResponse(
            translated_script=translated_script,
            original_script=request.script,
            language=request.target_language
        )
    
    # Fallback to single text
    if request.text:
        translated = translate_script(request.text, request.target_language)
        return TranslationResponse(
            translated_text=translated,
            original_text=request.text,
            language=request.target_language
        )
        
    return TranslationResponse(language=request.target_language) # Empty if nothing provided

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)