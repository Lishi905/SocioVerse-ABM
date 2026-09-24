"""
Configuration for Schelling segregation model with LLM integration.
"""

import os
from typing import Optional, Dict, Any

pwd_path = os.path.dirname(os.path.abspath(__file__))

class LLMConfig:
    """Configuration for LLM decision making."""
    
    # LLM Provider: "openai", "anthropic", "ollama"
    PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    
    # Model names for different providers
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # API Keys (from environment variables)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")  # read from the environment
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # LLM Parameters
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    USE_CACHE: bool = os.getenv("LLM_USE_CACHE", "true").lower() == "true"
    CACHE_DIR: str = os.getenv("LLM_CACHE_DIR", f"{pwd_path}/llm_cache")
    
    # Rate limiting (to avoid API quota issues) - increased for better performance
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_CALLS_PER_SECOND: float = float(os.getenv("RATE_LIMIT_CALLS_PER_SECOND", "10"))
    
    @classmethod
    def get_provider_config(cls) -> Dict[str, Any]:
        """Get configuration for the specified provider."""
        if cls.PROVIDER == "openai":
            return {
                "api_key": cls.OPENAI_API_KEY,
                "base_url": cls.OPENAI_BASE_URL,
                "model": cls.OPENAI_MODEL
            }
        elif cls.PROVIDER == "anthropic":
            return {
                "api_key": cls.ANTHROPIC_API_KEY,
                "model": cls.ANTHROPIC_MODEL
            }
        elif cls.PROVIDER == "ollama":
            return {
                "base_url": cls.OLLAMA_BASE_URL,
                "model": cls.OLLAMA_MODEL
            }
        else:
            raise ValueError(f"Unknown provider: {cls.PROVIDER}")


class SimulationConfig:
    """Configuration for Schelling segregation simulation."""
    
    # Grid parameters - smaller grid for faster simulation
    GRID_WIDTH: int = int(os.getenv("GRID_WIDTH", "15"))
    GRID_HEIGHT: int = int(os.getenv("GRID_HEIGHT", "15"))

    # Population parameters - higher density and minority % for better segregation effects
    DENSITY: float = float(os.getenv("DENSITY", "0.8"))
    MINORITY_PERCENTAGE: float = float(os.getenv("MINORITY_PERCENTAGE", "0.3"))
    HOMOPHILY: int = int(os.getenv("HOMOPHILY", "3"))

    # Simulation parameters - fewer steps for faster execution
    STEPS: int = int(os.getenv("STEPS", "10"))
    
    # Experiment parameters
    USE_LLM: bool = os.getenv("USE_LLM", "false").lower() == "true"
    LLM_ONLY_PERCENTAGE: float = float(os.getenv("LLM_ONLY_PERCENTAGE", "1.0"))
    
    # Output parameters
    SAVE_GIF: bool = os.getenv("SAVE_GIF", "true").lower() == "true"
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", f"{pwd_path}/output_figs")
    SAVE_EVERY_N_STEPS: int = int(os.getenv("SAVE_EVERY_N_STEPS", "1"))


class ExperimentConfig:
    """Configuration for running experiments."""
    
    # Experiment types
    EXPERIMENT_TYPES: list = [
        "llm_all",                   # All LLM-based agents
        "rule_based_only",           # Only rule-based agents
        "llm_hybrid_50",             # 50% LLM, 50% rule-based
        "llm_hybrid_mixed_types",    # Different types use different methods
    ]
    
    # Default experiment
    DEFAULT_EXPERIMENT: str = "llm_hybrid_50"
    
    @classmethod
    def get_experiment_config(cls, experiment_type: str) -> Dict[str, Any]:
        """Get configuration for specific experiment."""
        configs = {
            "rule_based_only": {
                "use_llm": False,
                "llm_only_percentage": 0.0
            },
            "llm_all": {
                "use_llm": True,
                "llm_only_percentage": 1.0
            },
            "llm_hybrid_50": {
                "use_llm": True,
                "llm_only_percentage": 0.5
            },
            "llm_hybrid_mixed_types": {
                "use_llm": True,
                "llm_only_percentage": 1.0  # Type 0 uses LLM, Type 1 uses rule-based
            }
        }
        
        if experiment_type not in configs:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        return configs[experiment_type]


# Print configuration on import (for debugging)
def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("LLM Configuration")
    print("=" * 60)
    print(f"Provider: {LLMConfig.PROVIDER}")
    print(f"Temperature: {LLMConfig.TEMPERATURE}")
    print(f"Cache Enabled: {LLMConfig.USE_CACHE}")
    print(f"Rate Limit Enabled: {LLMConfig.RATE_LIMIT_ENABLED}")
    print()
    print("=" * 60)
    print("Simulation Configuration")
    print("=" * 60)
    print(f"Grid Size: {SimulationConfig.GRID_WIDTH}x{SimulationConfig.GRID_HEIGHT}")
    print(f"Density: {SimulationConfig.DENSITY}")
    print(f"Minority %: {SimulationConfig.MINORITY_PERCENTAGE}")
    print(f"Homophily: {SimulationConfig.HOMOPHILY}")
    print(f"Steps: {SimulationConfig.STEPS}")
    print(f"Use LLM: {SimulationConfig.USE_LLM}")
    print(f"LLM Percentage: {SimulationConfig.LLM_ONLY_PERCENTAGE}")
    print()
