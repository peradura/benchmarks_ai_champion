# Model configurations for LongBench evaluation
# Add your models here with their specifications

MODEL_CONFIG = {
    # Qwen models
    "Qwen3-1.7B-Instruct": {
        "model_path": "Qwen/Qwen3-1.7B-Instruct",
        "max_model_len": 32768
    },
    "Qwen/Qwen3-1.7B-Instruct": {
        "model_path": "Qwen/Qwen3-1.7B-Instruct", 
        "max_model_len": 32768
    },
    "Qwen2.5-7B-Instruct": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "max_model_len": 131072  
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "max_model_len": 131072
    },
    
    # Llama models
    "Llama-3.1-8B-Instruct": {
        "model_path": "meta-llama/Llama-3.1-8B-Instruct",
        "max_model_len": 131072
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "model_path": "meta-llama/Llama-3.1-8B-Instruct", 
        "max_model_len": 131072
    },
    
    # Mistral models
    "Mistral-7B-Instruct-v0.3": {
        "model_path": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_model_len": 32768
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "model_path": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_model_len": 32768  
    },
    
    # Gemma models
    "gemma-2-2b-it": {
        "model_path": "google/gemma-2-2b-it",
        "max_model_len": 8192
    },
    "google/gemma-2-2b-it": {
        "model_path": "google/gemma-2-2b-it",
        "max_model_len": 8192
    },
    
    # Add more models as needed
    # Template:
    # "model_name": {
    #     "model_path": "huggingface/model_path",
    #     "max_model_len": context_length
    # }
}

def get_model_info(model_name):
    """Get model configuration"""
    return MODEL_CONFIG.get(model_name, {
        "model_path": model_name,
        "max_model_len": 32768  # Default
    })