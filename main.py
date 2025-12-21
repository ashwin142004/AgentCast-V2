from fastapi import FastAPI, HTTPException
from app.schemas import PodcastRequest, PodcastResponse
from app.workflow import build_graph

app = FastAPI(title="AgentCast V2 API")
graph = build_graph()

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
        "target_language": request.language,
        "final_script": ""
    }
    
    try:
        # Run graph to completion
        result = graph.invoke(initial_state)
        
        return PodcastResponse(
            status="completed",
            script=result["final_script"],
            original_script=None, # Could capture intermediate if needed
            language=request.language
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
