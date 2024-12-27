import time
from functools import wraps


def monitor_prediction_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Prediction time: {elapsed_time:.4f} seconds")
        return result
    return wrapper