"""
This component implement the basic functions powered by current LLMs for NaSch traffic model.
"""

from openai import OpenAI
import os
import re
import json

DEFAULT_BASE_URL = "https://api.openai.com/v1"
LLM_PROVIDERS = {
    "gpt-4o": "gpt-4o",
    "deepseek-v3": "deepseek-v3",
    "qwen3-235b": "qwen3-235b-a22b-instruct-2507",
}


def _get_api_key():
    """Get the OpenAI API key from the environment."""
    return os.environ.get("OPENAI_API_KEY")

# Cache for client (will be recreated if API key/base URL changes)
_client_cache = None
_cached_api_key = None
_cached_base_url = None

def _get_client():
    """Get OpenAI client, re-reading API key and base URL from environment each time"""
    global _client_cache, _cached_api_key, _cached_base_url
    
    # Always re-read from environment (in case it was set after module import)
    openai_key = _get_api_key()
    openai_base = os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    
    # Recreate client if API key or base URL changed
    if (_client_cache is None or 
        _cached_api_key != openai_key or 
        _cached_base_url != openai_base):
        
        if not openai_key:
            error_msg = (
                "OPENAI_API_KEY not found.\n"
                "PowerShell: $env:OPENAI_API_KEY=\"your-key-here\"\n"
                "Linux/Mac: export OPENAI_API_KEY=\"your-key-here\""
            )
            raise ValueError(error_msg)
        
        # Add timeout to prevent hanging - use a tuple for (connect, read) timeouts
        # This ensures both connection and read operations timeout properly
        # Reduced timeouts: 5s connect, 30s read (fail faster)
        try:
            _client_cache = OpenAI(
                api_key=openai_key, 
                base_url=openai_base,
                timeout=(5.0, 30.0)  # (connect timeout, read timeout) in seconds
            )
            _cached_api_key = openai_key
            _cached_base_url = openai_base
        except Exception as e:
            raise RuntimeError(f"Failed to create OpenAI client: {e}. Please check your API key and network connection.")
    
    return _client_cache


def generate(model, prompt, sys_prompt=None, max_tokens=16, temperature=0.1):
    if sys_prompt is None:
        sys_prompt = """You are a driver on a single-lane circular road.

To take a valid action, you MUST compute your speed according to these rules:
1. Acceleration: v_temp = min(v_current + 1, vmax)
2. Safety braking: v_safe = min(v_temp, gap_ahead)
3. Random hesitation: with probability p, reduce by 1: max(v_safe - 1, 0)

Any output outside the integer range [0, vmax] is an invalid action.
You MUST respond with ONLY a single integer. No explanation, no other text."""
    client = _get_client()
    try:
        import time as time_module
        start_time = time_module.time()
        create_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
        }
        response = client.chat.completions.create(**create_kwargs)
        elapsed = time_module.time() - start_time
        if elapsed > 5.0:
            print(f"[LLM API] API call took {elapsed:.2f}s (model: {model})")
        if not response or not response.choices:
            raise RuntimeError("OpenAI API returned empty response")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("OpenAI API returned empty content")
        return content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e


def safe_parse_speed_with_strategy(text, vmax=5, default=0):
    """
    Parse a single integer speed from LLM response.
    Strategy: find the LAST integer in the response that's in [0, vmax].
    """
    try:
        all_ints = re.findall(r'\b(\d+)\b', str(text).strip())
        valid = [int(x) for x in all_ints if 0 <= int(x) <= vmax]
        if valid:
            return valid[-1], "last_valid_integer"
        if all_ints:
            val = int(all_ints[-1])
            return max(0, min(val, vmax)), "clamped_last_integer"
    except Exception:
        pass
    return default, "default"


def safe_parse_speed(text, vmax=5, default=0):
    """Parse a single integer speed from text. Return default on failure."""
    value, _strategy = safe_parse_speed_with_strategy(text, vmax=vmax, default=default)
    return value


def safe_parse_number(text, default=0):
    """Backward-compatible numeric parser; speed parsing should use safe_parse_speed."""
    return safe_parse_speed(text, default=default)


def extract_json_from_gpt(content, model, vmax=5):
    """Parse a single speed integer from any LLM response; never raises."""
    value, strategy = safe_parse_speed_with_strategy(content, vmax=vmax, default=0)
    return {
        "speed": int(value),
        "raw_response": str(content),
        "parsed_speed": int(value),
        "parse_strategy": strategy,
    }


def generate_and_parser(model, prompt, sys_prompt=None, max_tokens=16, temperature=0.1):
    max_speed_match = re.search(r'vmax\s*=\s*(\d+)|max_speed\s*:\s*(\d+)', prompt)
    vmax = int(next(g for g in max_speed_match.groups() if g is not None)) if max_speed_match else 5
    max_times = 3
    count_time = 0
    while True:
        try:
            response = generate(model, prompt, sys_prompt, max_tokens, temperature)
            format_response = extract_json_from_gpt(response, model, vmax=vmax)
            v_llm = format_response.get("speed", 0)
            if v_llm is None or v_llm < 0:
                v_llm = 0
            elif v_llm > vmax:
                v_llm = vmax
            format_response["speed"] = int(v_llm)
            format_response["parsed_speed"] = int(v_llm)
            return format_response
        except Exception as e:
            count_time += 1
            if count_time >= max_times:
                print(f"[NaSch parse/API warning] default speed=0 after error: {e}")
                return {
                    "speed": 0,
                    "raw_response": "",
                    "parsed_speed": 0,
                    "parse_strategy": "api_error_default",
                    "error": str(e),
                }

def get_embedding(model, texts, embed_dim=3072):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, list):
        texts = texts
    else:
        raise TypeError("`texts` must be a string or a list of strings")
    
    client = _get_client()
    response = client.embeddings.create(
        model=model,
        input=texts,
        dimensions=embed_dim
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings


def parse_attributes(scenario, attributes):
    """
    Parse attributes and generate prompt for NaSch traffic model.
    
    Args:
        scenario: "nasch" for traffic model
        attributes: dict containing:
            - current_speed: int, current vehicle speed
            - gap_ahead: int, distance to next vehicle
            - max_speed: int, maximum allowed speed
            - randomization_prob: float, probability of random slowdown
    
    Returns:
        str: Formatted prompt for LLM
    """
    if scenario == "nasch":
        prompt = f"""Given v_current={attributes['current_speed']}, gap_ahead={attributes['gap_ahead']}, vmax={attributes['max_speed']}, p={attributes['randomization_prob']:.6f},
my speed is:"""
        return prompt

    else:
        raise ValueError(f"Unknown scenario: {scenario}")
