import requests
import json
import concurrent.futures
import time
import datetime

# API endpoint
url = "http://127.0.0.1:5002/v1/steer/completion"

# Headers
headers = {
    "Content-Type": "application/json",
    "x-secret-key": "cat"
}

# First request with "STEERED" type
data_steered = {
    "prompt": "hi",
    "steer_method": "SIMPLE_ADDITIVE",
    "normalize_steering": False,
    "model": "gpt2-small",
    "features": [
        {
            "model": "gpt2-small",
            "source": "7-res-jb",
            "index": 16899,
            "strength": 50
        }
    ],
    "strength_multiplier": 4,
    "n_completion_tokens": 5,
    "temperature": 1.0,
    "freq_penalty": 1.0,
    "seed": 32,
    "types": [
        "STEERED"
    ]
}

# Second request with "DEFAULT" type
data_default = {
    "prompt": "hi",
    "steer_method": "SIMPLE_ADDITIVE",
    "normalize_steering": False,
    "model": "gpt2-small",
    "features": [
        {
            "model": "gpt2-small",
            "source": "7-res-jb",
            "index": 16899,
            "strength": 50
        }
    ],
    "strength_multiplier": 4,
    "n_completion_tokens": 5,
    "temperature": 1.0,
    "freq_penalty": 1.0,
    "seed": 32,
    "types": [
        "DEFAULT"
    ]
}

def send_request(data, type_name):
    start_time = time.time()
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n[{current_time}] --- {type_name} Request STARTED ---")
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        end_time = time.time()
        duration = end_time - start_time
        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        print(f"[{current_time}] --- {type_name} Response RECEIVED ---")
        print(f"Duration: {duration:.3f} seconds")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response, duration
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{current_time}] Error with {type_name} request: {e}")
        print(f"Duration until error: {duration:.3f} seconds")
        return None, duration

# Print overall start time
overall_start = time.time()
print(f"\n===== TESTING SEQUENTIAL VS PARALLEL PROCESSING =====")
print(f"Starting both requests at: {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

# Send both requests in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_steered = executor.submit(send_request, data_steered, "STEERED")
    future_default = executor.submit(send_request, data_default, "DEFAULT")
    
    # Wait for both to complete
    steered_response, steered_duration = future_steered.result()
    default_response, default_duration = future_default.result()

overall_end = time.time()
overall_duration = overall_end - overall_start

print("\n===== TIMING SUMMARY =====")
print(f"Total time for both requests: {overall_duration:.3f} seconds")
print(f"STEERED request duration: {steered_duration:.3f} seconds")
print(f"DEFAULT request duration: {default_duration:.3f} seconds")
print(f"Sum of individual times: {steered_duration + default_duration:.3f} seconds")
print(f"Overall duration: {overall_duration:.3f} seconds")
# Analyze if they ran sequentially or in parallel
if overall_duration < (steered_duration + default_duration) * 0.9:  # Allow for some timing variance
    print("\nRequests appear to have run IN PARALLEL (total time < sum of individual times)")
else:
    print("\nRequests appear to have run SEQUENTIALLY (total time ≈ sum of individual times)")
    print(f"Evidence: Total time ({overall_duration:.3f}s) is approximately equal to the sum of individual times ({steered_duration + default_duration:.3f}s)")