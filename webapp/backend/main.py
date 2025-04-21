from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .tasks import run_metaheuristic_task
from fastapi.staticfiles import StaticFiles
from celery.result import AsyncResult
import redis
import json

# Initialize Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

app = FastAPI()
app.mount("/static", StaticFiles(directory="webapp/frontend"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizationRequest(BaseModel):
    algorithm: str
    num_bs: int
    num_ue: int
    ue_positions: list[list[float]] = None  # Add position support

@app.get("/")
def root():
    return {"message": "Metaheuristic backend is running!"}

@app.post("/start")
async def start_optimization(request: OptimizationRequest):
    """Endpoint to start optimization"""
    try:
        # Validate UE positions
        if request.ue_positions:
            if len(request.ue_positions) != request.num_ue:
                raise HTTPException(status_code=400, detail="Mismatch between num_ue and positions count")
                
        task = run_metaheuristic_task.delay(request.model_dump())
        return {"task_id": task.id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Check task progress with positions"""
    task = AsyncResult(task_id)
    
    if task.state == 'FAILURE':
        return {"status": "error", "message": str(task.result)}
    
    # Get positions from Redis
    positions = redis_client.get(f"positions:{task_id}")
    
    response = {
        "status": task.state,
        "result": None
    }
    
    if task.ready():
        response["result"] = {
            "solution": task.result.get("solution", []),
            "metrics": task.result.get("metrics", {}),
            "history": task.result.get("history", {})
        }
        
    return {
        "status": task.state,
        "result": task.result if task.ready() else None,
        "progress": task.info.get("data") if task.info else None,
        "positions": json.loads(positions) if positions else []
    }

@app.get("/positions/{task_id}")
async def get_positions(task_id: str):
    """Direct position endpoint for Streamlit polling"""
    positions = redis_client.get(f"positions:{task_id}")
    return json.loads(positions) if positions else []
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from .tasks import run_metaheuristic_task
# from fastapi.staticfiles import StaticFiles

# app = FastAPI()
# app.mount("/static", StaticFiles(directory="webapp/frontend"), name="static")

# # Allow frontend requests (replace with your frontend URL in production)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class OptimizationRequest(BaseModel):
#     algorithm: str
#     num_bs: int
#     num_ue: int # [[x1,y1], [x2,y2], ...]
    
# @app.get("/")
# def root():
#     return {"message": "Metaheuristic backend is running!"}


# @app.post("/start")
# async def start_optimization(request: OptimizationRequest):
#     """Endpoint to start optimization"""
#     task = run_metaheuristic_task.delay(request.dict())
#     return {"task_id": task.id}

# @app.get("/status/{task_id}")
# async def get_status(task_id: str):
#     """Check task progress"""
#     task = run_metaheuristic_task.AsyncResult(task_id)
    
#     if task.state == 'FAILURE':
#         return {"status": "error", "message": str(task.result)}
    
#     return {
#         "status": task.state,
#         "result": task.result if task.ready() else None,
#         "progress": task.info.get("data") if task.info else None
#     }