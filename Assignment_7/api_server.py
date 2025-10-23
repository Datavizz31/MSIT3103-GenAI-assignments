from fastapi import FastAPI, Request
from pydantic import BaseModel
from gen_ai_model import SimpleGenerativeAI
import time
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# Load the model once at startup
gen_ai = SimpleGenerativeAI("gpt2")

# Enable CORS so the React frontend can POST to the FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to ["http://localhost:5173"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the frontend if dist directory exists
frontend_dist_path = "frontend/dist"
if os.path.isdir(frontend_dist_path):
    app.mount("/assets", StaticFiles(directory=f"{frontend_dist_path}/assets"), name="assets")

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 80
    temperature: float = 0.5
    num_return_sequences: int = 1

@app.post("/generate/")
def generate_text(request: GenerationRequest):
    start = time.time()
    outputs = gen_ai.generate_text(
        prompt=request.prompt,
        max_length=request.max_length,
        temperature=request.temperature,
        num_return_sequences=request.num_return_sequences
    )
    latency = time.time() - start
    return {
        "outputs": outputs,
        "latency_seconds": latency
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# Serve the frontend for all other routes (SPA)
@app.get("/{path:path}")
def serve_frontend(path: str):
    frontend_index = "frontend/dist/index.html"
    if os.path.isfile(frontend_index):
        return FileResponse(frontend_index)
    # Fallback if frontend not built
    return {"message": "Frontend not built. Run 'npm run build' in the frontend directory."}

@app.get("/")
def root():
    frontend_index = "frontend/dist/index.html"
    if os.path.isfile(frontend_index):
        return FileResponse(frontend_index)
    return {
        "message": "Simple Generative AI API",
        "endpoints": {
            "health": "GET /health",
            "generate": "POST /generate/",
            "docs": "GET /docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
