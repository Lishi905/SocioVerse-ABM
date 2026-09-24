# NOTE (SocioVerse-ABM public release): legacy snapshot kept for provenance; only the LLM
# base-URL default was changed. Not maintained (see the README in this legacy folder).
# Comments and messages may be in the original authors' language (Chinese).
"""
LLM Integration module for Schelling segregation agents.
Supports multiple LLM providers (OpenAI, Anthropic, Ollama, etc.)
"""

import os
import json
import time
import asyncio
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from functools import lru_cache
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def query(self, prompt: str, **kwargs) -> str:
        """Query the LLM and return response."""
        pass
    
    @abstractmethod
    async def query_async(self, prompt: str, **kwargs) -> str:
        """Async query the LLM and return response."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1", model: str = "gpt-3.5-turbo"):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)
            self.model = model
            self.available = True
        except ImportError:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
            self.available = False
    
    def query(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI query failed: {e}")
            raise
    
    async def query_async(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Async query for OpenAI (uses asyncio to avoid blocking)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.query,
            prompt,
            temperature
        )
    
    def is_available(self) -> bool:
        return self.available


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = model
            self.available = True
        except ImportError:
            logger.warning("Anthropic package not installed. Install with: pip install anthropic")
            self.available = False
    
    def query(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic query failed: {e}")
            raise
    
    async def query_async(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Async query for Anthropic (uses asyncio to avoid blocking)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.query,
            prompt,
            temperature
        )
    
    def is_available(self) -> bool:
        return self.available


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        try:
            import requests
            self.base_url = base_url
            self.model = model
            self.available = requests.get(f"{base_url}/api/tags").status_code == 200
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}")
            self.available = False
    
    def query(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                }
            )
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
            raise
    
    async def query_async(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Async query for Ollama using aiohttp."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "stream": False
                    }
                ) as resp:
                    data = await resp.json()
                    return data["response"]
        except ImportError:
            # Fallback to blocking query if aiohttp not available
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.query, prompt, temperature)
        except Exception as e:
            logger.error(f"Ollama async query failed: {e}")
            raise
    
    def is_available(self) -> bool:
        return self.available


class LLMCache:
    """Simple cache for LLM responses to avoid repeated queries."""
    
    def __init__(self, cache_dir: str = "./llm_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _hash_prompt(self, prompt: str) -> str:
        """Generate hash of prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[str]:
        """Get cached response if exists."""
        cache_file = os.path.join(self.cache_dir, f"{self._hash_prompt(prompt)}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get("response")
            except:
                pass
        return None
    
    def set(self, prompt: str, response: str):
        """Cache LLM response."""
        cache_file = os.path.join(self.cache_dir, f"{self._hash_prompt(prompt)}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump({"prompt": prompt, "response": response}, f)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")


class LLMDecisionMaker:
    """Main interface for LLM-based decision making in Schelling model."""
    
    def __init__(self, provider_name: str = "openai", use_cache: bool = True, **kwargs):
        """
        Initialize LLM decision maker.
        
        Args:
            provider_name: "openai", "anthropic", or "ollama"
            use_cache: Whether to cache LLM responses
            **kwargs: Additional arguments passed to provider
        """
        self.provider_name = provider_name.lower()
        self.cache = LLMCache() if use_cache else None
        self.use_cache = use_cache
        
        # Initialize provider
        if self.provider_name == "openai":
            self.provider = OpenAIProvider(**kwargs)
        elif self.provider_name == "anthropic":
            self.provider = AnthropicProvider(**kwargs)
        elif self.provider_name == "ollama":
            self.provider = OllamaProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        if not self.provider.is_available():
            raise RuntimeError(f"Provider {provider_name} is not available")
        
        logger.info(f"LLM Decision Maker initialized with provider: {provider_name}")
    
    async def make_step_decision_async(self, 
                                      agent_pos: tuple,
                                      agent_type: int,
                                      neighbors_info: Dict[str, Any],
                                      empty_cells: List[tuple],
                                      grid_size: tuple,
                                      homophily_threshold: int,
                                      **kwargs) -> Dict[str, Any]:
        """
        Async version of make_step_decision using asyncio for concurrent execution.
        
        Use this in async contexts for better performance with many agents.
        """
        prompt = self._construct_step_prompt(
            agent_pos, agent_type, neighbors_info, empty_cells, 
            grid_size, homophily_threshold
        )
        
        # Check cache
        if self.use_cache:
            cached_response = self.cache.get(prompt)
            if cached_response:
                logger.debug("Cache hit for step decision (async)")
                return self._parse_step_decision(cached_response, empty_cells)
        
        # Query LLM asynchronously
        try:
            response = await self.provider.query_async(prompt, **kwargs)
            
            # Cache response
            if self.use_cache:
                self.cache.set(prompt, response)
            
            decision = self._parse_step_decision(response, empty_cells)
            logger.debug(f"LLM async step decision: {decision['action']}")
            return decision
        
        except Exception as e:
            logger.error(f"LLM async step decision failed: {e}")
            # Fallback to rule-based decision
            return self._rule_based_step_decision(
                agent_pos, neighbors_info, empty_cells, homophily_threshold
            )
    
    async def make_batch_decisions_async(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make decisions for multiple agents concurrently using asyncio.
        
        This is significantly faster than sequential calls when using LLM.
        
        Args:
            batch_requests: List of decision request dicts, each containing:
                - agent_pos: Current position (x, y)
                - agent_type: Agent's type (0 or 1)
                - neighbors_info: Dict with neighbor information
                - empty_cells: List of available positions
                - grid_size: Tuple of (width, height)
                - homophily_threshold: Minimum similar neighbors for happiness
        
        Returns:
            List of decision dicts in same order as input
        """
        tasks = []
        
        for request in batch_requests:
            task = self.make_step_decision_async(
                agent_pos=request["agent_pos"],
                agent_type=request["agent_type"],
                neighbors_info=request["neighbors_info"],
                empty_cells=request["empty_cells"],
                grid_size=request["grid_size"],
                homophily_threshold=request["homophily_threshold"],
                **request.get("kwargs", {})
            )
            tasks.append(task)
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
                processed_results.append({
                    "action": "stay",
                    "new_pos": None,
                    "reasoning": "Error in async decision",
                    "reasoning_factors": [],
                    "confidence": 0.0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def make_step_decision(self, 
                          agent_pos: tuple,
                          agent_type: int,
                          neighbors_info: Dict[str, Any],
                          empty_cells: List[tuple],
                          grid_size: tuple,
                          homophily_threshold: int,
                          **kwargs) -> Dict[str, Any]:
        """
        Use LLM to make complete step decision including:
        1. Whether to move or stay
        2. If move, where to go (specific location)
        3. Reasoning and considerations
        
        Args:
            agent_pos: Current position (x, y)
            agent_type: Agent's type (0 or 1)
            neighbors_info: Dict with:
                - similar_neighbors: Number of similar type neighbors
                - total_neighbors: Total number of neighbors
                - neighbor_types: List of neighbor types
            empty_cells: List of available empty cell positions
            grid_size: Tuple of (width, height)
            homophily_threshold: Minimum similar neighbors for happiness
            **kwargs: Additional context
        
        Returns:
            Dict with:
            - "action": "stay" or "move"
            - "new_pos": New position if moving, None if staying
            - "reasoning": LLM's explanation
            - "reasoning_factors": Key factors considered
        """
        # Construct rich prompt
        prompt = self._construct_step_prompt(
            agent_pos, agent_type, neighbors_info, empty_cells, 
            grid_size, homophily_threshold
        )
        
        # Check cache
        if self.use_cache:
            cached_response = self.cache.get(prompt)
            if cached_response:
                logger.debug("Cache hit for step decision")
                return self._parse_step_decision(cached_response, empty_cells)
        
        # Query LLM
        try:
            response = self.provider.query(prompt, **kwargs)
            
            # Cache response
            if self.use_cache:
                self.cache.set(prompt, response)
            
            decision = self._parse_step_decision(response, empty_cells)
            logger.debug(f"LLM step decision: {decision['action']}, response: {response[:150]}")
            return decision
        
        except Exception as e:
            logger.error(f"LLM step decision failed: {e}")
            # Fallback to rule-based decision
            return self._rule_based_step_decision(
                agent_pos, neighbors_info, empty_cells, homophily_threshold
            )
    
    def _construct_step_prompt(self,
                              agent_pos: tuple,
                              agent_type: int,
                              neighbors_info: Dict,
                              empty_cells: List[tuple],
                              grid_size: tuple,
                              homophily_threshold: int) -> str:
        """Construct comprehensive prompt for complete step decision."""
        
        similar = neighbors_info.get("similar_neighbors", 0)
        total = neighbors_info.get("total_neighbors", 0)
        neighbor_types = neighbors_info.get("neighbor_types", [])
        
        # Analyze neighborhood composition
        similarity_pct = 100 * similar / max(1, total)
        is_happy = similar >= homophily_threshold
        
        # Sample empty cells to show (too many to list all)
        sample_empty = empty_cells[:min(5, len(empty_cells))]
        
        prompt = f"""你是一个社区模拟中的理性居民代理。现在需要决定这一步的行动。

【你的当前位置信息】
- 当前位置: ({agent_pos[0]}, {agent_pos[1]})
- 你的身份: 类型{agent_type}的居民

【你的邻居情况】
- 与你相同类型的邻居: {similar}个
- 总邻居数: {total}个
- 相似性比例: {similarity_pct:.1f}%
- 邻居类型分布: {neighbor_types}

【社区信息】
- 网格大小: {grid_size[0]} × {grid_size[1]}
- 总共可用的空房间: {len(empty_cells)}个
- 示例空房间位置(前5个): {sample_empty}

【你的偏好设置】
- 你的容忍度阈值: 需要至少{homophily_threshold}个相同类型邻居才能感到满意
- 你当前的满意度: {'满意 ✓' if is_happy else '不满意 ✗'}

【需要做出的决策】
请基于以下因素进行深思熟虑:
1. 邻居多样性: 是否能接受当前的邻居组成?
2. 距离成本: 搬家需要时间和精力成本
3. 不确定性: 新位置的邻居情况未知
4. 社会稳定性: 频繁搬家会影响社区稳定
5. 长期收益: 新位置是否值得搬家的努力?

请以JSON格式返回你的完整决策:
{{
  "decision": "stay" | "move",
  "target_position": (x, y) | null,
  "confidence": 0.0-1.0,
  "reasoning": "详细解释你的决策过程",
  "factors_considered": ["因素1", "因素2", "..."],
  "alternatives_rejected": ["为什么不选择其他选项"]
}}

重要: 
- 只返回有效的JSON,不要其他文本
- 如果decision为"move",target_position必须是一个现有的空房间坐标
- confidence反映你对这个决策的确定程度(0=非常不确定, 1=完全确定)
- 要充分考虑搬家的成本与收益平衡"""
        
        return prompt
    
    def _parse_step_decision(self, response: str, empty_cells: List[tuple]) -> Dict[str, Any]:
        """Parse LLM response to extract complete step decision."""
        try:
            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # Parse decision
                decision = data.get("decision", "stay").lower()
                target_pos = None
                
                if decision == "move":
                    target_tuple = data.get("target_position")
                    if target_tuple and isinstance(target_tuple, (list, tuple)):
                        target_pos = tuple(target_tuple)
                        # Validate target position
                        if target_pos not in empty_cells:
                            # Fallback: pick a random empty cell
                            target_pos = empty_cells[0] if empty_cells else None
                
                return {
                    "action": decision,
                    "new_pos": target_pos,
                    "reasoning": data.get("reasoning", "No reasoning provided"),
                    "reasoning_factors": data.get("factors_considered", []),
                    "confidence": float(data.get("confidence", 0.5))
                }
        except Exception as e:
            logger.debug(f"Failed to parse JSON decision: {e}")
        
        # Fallback: keyword-based parsing
        response_lower = response.lower()
        
        # Try to extract coordinates if mentioned
        import re
        coords_match = re.search(r'\((\d+)\s*,\s*(\d+)\)', response)
        target_pos = None
        if coords_match and empty_cells:
            try:
                x, y = int(coords_match.group(1)), int(coords_match.group(2))
                if (x, y) in empty_cells:
                    target_pos = (x, y)
            except:
                pass
        
        # Determine action
        if any(word in response_lower for word in ["搬家", "move", "应该移动", "prefer to move"]):
            action = "move"
            if not target_pos and empty_cells:
                target_pos = empty_cells[0]
        else:
            action = "stay"
        
        return {
            "action": action,
            "new_pos": target_pos,
            "reasoning": response[:200],
            "reasoning_factors": ["Unable to parse factors"],
            "confidence": 0.3  # Low confidence for keyword parsing
        }
    
    def _rule_based_step_decision(self, 
                                  agent_pos: tuple,
                                  neighbors_info: Dict,
                                  empty_cells: List[tuple],
                                  homophily_threshold: int) -> Dict[str, Any]:
        """Fallback to traditional rule-based step decision."""
        similar = neighbors_info.get("similar_neighbors", 0)
        
        should_move = similar < homophily_threshold
        new_pos = None
        
        if should_move and empty_cells:
            import random
            new_pos = random.choice(empty_cells)
        
        return {
            "action": "move" if should_move else "stay",
            "new_pos": new_pos,
            "reasoning": f"Rule-based: {'unhappy' if should_move else 'happy'} with {similar} similar neighbors (threshold: {homophily_threshold})",
            "reasoning_factors": ["similar_neighbors", "homophily_threshold"],
            "confidence": 0.8  # High confidence for deterministic rule
        }
    
    def make_decision(self, 
                      agent_type: int, 
                      similar_neighbors: int, 
                      total_neighbors: int,
                      homophily_threshold: int,
                      empty_cells_available: int,
                      **kwargs) -> bool:
        """
        Legacy method for backward compatibility.
        Use make_step_decision() for new implementations.
        
        Returns:
            True if agent should move, False otherwise
        """
        return similar_neighbors < homophily_threshold
    
    def _construct_prompt(self,
                          agent_type: int,
                          similar_neighbors: int,
                          total_neighbors: int,
                          homophily_threshold: int,
                          empty_cells_available: int) -> str:
        """Legacy prompt construction for backward compatibility."""
        prompt = f"""你是一个城市规划模拟中的居民代理。
        
你的身份: 类型{agent_type}的居民
你的邻居情况:
- 与你相同类型的邻居: {similar_neighbors}个
- 总邻居数: {total_neighbors}个
- 相似性比例: {similar_neighbors}/{total_neighbors} = {100*similar_neighbors/max(1, total_neighbors):.1f}%

环境信息:
- 你对邻居的容忍度阈值: 至少需要{homophily_threshold}个相同类型的邻居
- 当前可用的空房间数: {empty_cells_available}个

根据这些信息,你需要决定:
1. 你是否对当前位置感到满意(即相同类型邻居数 >= {homophily_threshold})?
2. 如果不满意,你是否应该搬家?

请以JSON格式回答:
{{"should_move": true/false, "reason": "简要说明你的理由"}}

只返回JSON,不要其他文本。"""
        return prompt
    
    def _parse_decision(self, response: str) -> bool:
        """Legacy decision parsing for backward compatibility."""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                return data.get("should_move", False)
        except:
            pass
        
        response_lower = response.lower()
        if "应该搬家" in response_lower or "should move" in response_lower or "yes" in response_lower:
            return True
        elif "不应该搬" in response_lower or "should not move" in response_lower or "no" in response_lower:
            return False
        
        return False
