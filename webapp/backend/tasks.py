from celery import Celery
from core.hybrid_trainer.metaheuristic_opt import run_metaheuristic, serialize_result
from core.envs.custom_channel_env import NetworkEnvironment
from core. hybrid_trainer.kpi_logger import WebKPILogger
import numpy as np
import os
import redis
import json

# Initialize Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    decode_responses=False
)

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
# In tasks.py
# celery.conf.update(
#     worker_pool='threads',  # Use threads instead of processes
#     broker_connection_retry_on_startup=True,
#     task_always_eager=False,
#     worker_disable_rate_limits=True
# )
# # Initialize Redis connection
# redis_client = redis.Redis(
#     host='localhost',
#     port=6379,
#     db=0,
#     decode_responses=False  # Keep as False for binary data storage
# )


@celery.task(bind=True)
def run_metaheuristic_task(self, config):
    """Celery task wrapper for your optimization"""
    try:
        env_config = {
            "num_bs": config.get("num_bs", 20),
            # Prioritize ue_positions if provided
            # "ue_positions": config.get("ue_positions"),
            "num_ue": config.get("num_ue")  # Fallback if ue_positions is missing
        }
        
        env = NetworkEnvironment(config=env_config, log_kpis=True)
        
        # Web-enabled logger
        web_logger = WebKPILogger(celery_task=self, enabled=True)
        
        
        result = run_metaheuristic(
            env=env,
            algorithm=config["algorithm"],
            epoch=0,
            kpi_logger=web_logger,
            visualize_callback=web_visualize  # Optional
            # visualize_callback=partial(web_visualize, task=self)  # New web viz
        )
        print(f"{result}")
        # serial_res=serialize_result(result)
        # print(serial_res)
        # return {"status": "success", "result": serialize_result(result)} 
        return {
                'status': 'success',
                "result": serialize_result(result), # Contains solution and metrics
                # 'solution': solution,
                # 'metrics': metrics,  # Directly return metrics dict
                'history': WebKPILogger.history.to_dict()  # Add history tracking
            } 
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
# def web_visualize(task, iteration, positions, fitness):
#     """Push visualization data to frontend"""
#     task.update_state(
#         state='PROGRESS',
#         meta={
#             'type': 'VISUALIZATION',
#             'iteration': iteration,
#             'positions': positions.tolist(),  # Convert numpy arrays
#             'fitness': float(np.mean(fitness))
#         }
#     )

def web_visualize(task, iteration: int, agents: dict, metrics: dict):
    """Store positions in Redis instead of WebSocket"""
    try:
        # Store positions with task ID
        print(f"Storing positions for iteration {iteration}")
        redis_client.set(
            f"positions:{task.id}",
            json.dumps({
                "iteration": iteration,
                "positions": agents["positions"],
                "metrics": {
                    "fitness": metrics.get("fitness", 0),
                    "sinr": metrics.get("average_sinr", 0)
                }
            }),
            ex=3600  # Expire after 1 hour
        )
    except Exception as e:
        print(f"Position storage error: {str(e)}")

# def run_metaheuristic_task(self, config: Dict) -> Dict:
#     """
#     Web-friendly version that handles:
#     - Progress tracking
#     - Async cancellation
#     - Result serialization
#     """
#     try:
#         # Initialize environment from web config
#         # env = NetworkEnvironment.from_config(config)
#         env = NetworkEnvironment(
#         num_bs=int(config["num_bs"]),
#         ue_positions=np.array(config["ue_positions"])  # List → numpy
#     )
        
#         # Initialize web-enabled logger
#         web_logger = WebKPILogger(
#             celery_task=self,  # Pass the Celery task reference
#             enabled=True
#         )
        
#         # Add progress reporting
#         self.update_state(state='PROGRESS', meta={'status': 'Initializing...'})
        
#         # Modified to pass web-friendly logger
#         result = run_metaheuristic(
#             env=env,
#             algorithm=config['algorithm'],
#             epoch=0,  # Web runs typically single epoch
#             kpi_logger=web_logger,
#             visualize_callback=partial(web_visualize, task=self)  # New web viz
#         )
        
#         return {
#             'status': 'SUCCESS',
#             'result': serialize_result(result)  # Convert numpy to native types
#         }
#     except Exception as e:
#         return {'status': 'FAILURE', 'error': str(e)}