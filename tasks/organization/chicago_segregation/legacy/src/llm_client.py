"""
LLM Client for Chicago Segregation ABM.

Handles archetype-level LLM calls for:
1. Satisfaction assessment: How satisfied is this archetype with their current neighborhood?
2. Move evaluation: Evaluate candidate neighborhoods for relocation.

Uses OpenAI-compatible API with structured JSON output.
Supports both synchronous and async (batched concurrent) call modes.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

SATISFACTION_SYSTEM_PROMPT = """\
You are simulating a household's residential satisfaction in Chicago (year 2010).
You must evaluate how satisfied this household is with their current neighborhood.

City-wide demographic context (2010 Chicago, public baseline): the overall
population is roughly 32% Non-Hispanic White, 32% Non-Hispanic Black, 29%
Hispanic, 5% Non-Hispanic Asian, and ~2% Other.  A tract's racial makeup
should be interpreted relative to this citywide baseline — e.g. a 15%
Asian tract is well above the citywide Asian share, so "limited Asian
presence" is not a reasonable complaint there.

Consider ALL of these factors — use the importance weights from the household profile:
- School quality and educational access
- Safety and crime levels
- Housing affordability relative to income
- Commute and transit access
- Local amenities (parks, grocery, religious places)
- Proximity to social network (family, friends, community)
- Neighborhood demographic familiarity (see household profile for details)

IMPORTANT — STAY BIAS: Moving is costly in real life: packing, transaction
costs, broken school continuity, lost neighborhood ties, and uncertainty
about the new area.  The default for an established household is to STAY.
Only set would_move=true when a TOP-weighted priority (weight ≥ 20%) is
badly unmet — typically scoring 4/10 or worse — AND it is plausible that
other reachable neighborhoods would do noticeably better on that priority
without sacrificing the household's other top priorities.  A household
that is mildly dissatisfied (satisfaction 5–6) on several factors but has
no single severely unmet top priority should keep would_move=false.
Racial/demographic discomfort counts as a legitimate driver only for
groups where it is explicitly described as an important, strongly-weighted
priority in the household profile.

Respond ONLY with valid JSON in this exact format:
{"satisfaction": <float 0-10>, "would_move": <bool>, "key_reasons": ["reason1", "reason2"]}

Where satisfaction 0 = extremely dissatisfied, 10 = perfectly satisfied.
would_move = true if this household would actively seek to relocate."""

MOVE_EVAL_SYSTEM_PROMPT = """\
You are simulating a household's neighborhood choice decision in Chicago (year 2010).
Evaluate candidate neighborhoods and pick the best one for relocation.

City-wide demographic context (2010 Chicago, public baseline): the overall
population is roughly 32% Non-Hispanic White, 32% Non-Hispanic Black, 29%
Hispanic, 5% Non-Hispanic Asian, and ~2% Other.  When judging
"demographic familiarity" in any candidate tract, compare to this
citywide baseline — do not treat the arithmetic most-same-race tract as
automatically the most desirable.  A tract that is 15% Asian is already
three times the citywide Asian share and offers a visible Asian
community; a 40% Asian tract is not meaningfully "more familiar" for
amenity purposes and will tend to be further from jobs and schools.

Moving imposes real costs (packing, transaction fees, broken school
continuity, lost neighborhood ties). The stay_score should already
reflect a modest "inertia premium" for the current neighborhood — a
candidate must be meaningfully better on a top priority to justify the
move, not merely marginally better.

Weigh ALL factors from the household profile — schools, safety, affordability,
commute, amenities, social networks, and demographic familiarity — according to
the stated importance weights.

Score the current neighborhood honestly based on how well it meets the household's
needs right now. Score candidate neighborhoods the same way. Do NOT apply a blanket
"familiarity bonus" — a household that is genuinely dissatisfied with their current
neighborhood should see their current stay_score reflect that dissatisfaction.

Recommend a move (best_score > stay_score) whenever a candidate is clearly better
on the household's top-weighted priorities. Even a meaningful improvement in ONE top
priority is a valid reason to move if other factors are roughly equal.

CRITICAL: If the household profile states that ethnic composition is a minor or
non-factor in their decision (e.g., Asian or Hispanic households), do NOT let
demographic composition tip the decision. For these households, only recommend
moving when schools, transit, safety, or affordability improve substantially.

HARD CONSTRAINT — anti-sorting rule (applies to ALL households, regardless of
race): If a candidate neighborhood has a HIGHER share of the household's OWN
race than the current neighborhood, AND none of the top-weighted priorities
(schools, safety, affordability, commute) is clearly better by at least 2 points
on the 10-point scale, you MUST set stay_score >= best_score so the household
stays. In other words: moving to a MORE same-race concentrated tract requires
concrete, substantial improvement on a non-demographic priority. This rule
prevents mechanical race-sorting toward same-race enclaves and reflects the
real cost of moving (packing, new schools, broken ties) which always outweighs
a marginal demographic comfort gain.

TIE-BREAKER RULE (applies when several candidates score within 1 point of each
other on non-demographic factors): prefer the candidate with the MORE DIVERSE
demographic mix (closer to 20–40% own-race share, not 70–90%).  Moderate
exposure to other races is a neutral-to-positive factor for most 2010 Chicago
households unless their profile explicitly demands a same-race environment
(strong-preference Black and White households).  Do NOT pick a candidate merely
because it has the highest own-race share in the filtered list — that is
exactly the mechanical sorting we want to avoid.

POSITIVE FRAMING: A tract with 25% own-race share, good schools, and safe
streets is a BETTER choice than a tract with 80% own-race share and mediocre
schools.  Mixed neighborhoods are normal and desirable for most households.

Respond ONLY with valid JSON in this exact format:
{"ranked_candidates": [{"tract_id": "<tract_id>", "score": <float 0-10>}, ...], "stay_score": <float 0-10>, "reason": "<one sentence>"}

ranked_candidates = a list of UP TO 3 candidate tracts sorted by your preference
(best first). Include fewer than 3 only if fewer candidates are genuinely
plausible. Each entry gives your attractiveness score (0-10) for that tract.
stay_score = how attractive staying in the current neighborhood is (0-10).
If stay_score >= the top candidate's score, the household should stay — in
that case return an empty ranked_candidates list."""


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 150
    min_call_interval: float = 0.0  # Minimum seconds between API calls (rate limiting)
    max_concurrent: int = 20  # Max concurrent async requests


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call for logging."""
    call_id: int
    call_type: str  # "satisfaction" or "move_eval"
    archetype_key: str = ""
    tract_id: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    raw_response: str = ""
    parsed_result: dict = field(default_factory=dict)
    is_fallback: bool = False
    fallback_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.async_client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_log: list[LLMCallRecord] = []
        self._last_call_time = 0.0

    def assess_satisfaction(self, archetype_desc: str, neighborhood_desc: str,
                            archetype_key: str = "", tract_id: str = "") -> dict:
        """
        Ask LLM to assess an archetype's satisfaction with their current neighborhood.

        Returns dict with keys: satisfaction (float), would_move (bool), key_reasons (list)
        """
        user_msg = f"""Household profile:
{archetype_desc}

Current neighborhood conditions:
{neighborhood_desc}

Evaluate this household's satisfaction with their current neighborhood."""

        return self._call(SATISFACTION_SYSTEM_PROMPT, user_msg,
                          call_type="satisfaction", archetype_key=archetype_key, tract_id=tract_id)

    def evaluate_move(self, archetype_desc: str, current_desc: str, candidates_desc: str,
                      archetype_key: str = "", tract_id: str = "") -> dict:
        """
        Ask LLM to pick the best neighborhood for relocation.

        Returns dict with keys: best_tract, best_score, stay_score, reason
        """
        user_msg = f"""Household profile:
{archetype_desc}

Current neighborhood:
{current_desc}

Candidate neighborhoods:
{candidates_desc}

Pick the best neighborhood to move to, or recommend staying."""

        return self._call(MOVE_EVAL_SYSTEM_PROMPT, user_msg,
                          call_type="move_eval", archetype_key=archetype_key, tract_id=tract_id)

    def _call(self, system_prompt: str, user_prompt: str,
              call_type: str = "", archetype_key: str = "", tract_id: str = "") -> dict:
        """Make an LLM API call and parse JSON response."""
        # Rate limiting
        if self.config.min_call_interval > 0:
            elapsed = time.time() - self._last_call_time
            if elapsed < self.config.min_call_interval:
                time.sleep(self.config.min_call_interval - elapsed)

        self.call_count += 1
        record = LLMCallRecord(
            call_id=self.call_count,
            call_type=call_type,
            archetype_key=archetype_key,
            tract_id=tract_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        start_time = time.time()
        self._last_call_time = start_time

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            record.latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content.strip()
            record.raw_response = content

            if response.usage:
                record.input_tokens = response.usage.prompt_tokens
                record.output_tokens = response.usage.completion_tokens
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens

            # Parse JSON - handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            record.parsed_result = parsed
            self.call_log.append(record)
            return parsed

        except json.JSONDecodeError as e:
            record.latency_ms = (time.time() - start_time) * 1000
            record.is_fallback = True
            record.fallback_reason = f"JSON parse error: {e}"
            record.parsed_result = {"satisfaction": 5.0, "would_move": False, "key_reasons": ["parse_error"]}
            self.call_log.append(record)
            logger.warning(f"LLM returned invalid JSON: {content[:200]}... Error: {e}")
            return record.parsed_result
        except Exception as e:
            record.latency_ms = (time.time() - start_time) * 1000
            record.is_fallback = True
            record.fallback_reason = f"API error: {e}"
            record.error = str(e)
            record.parsed_result = {"satisfaction": 5.0, "would_move": False, "key_reasons": ["api_error"]}
            self.call_log.append(record)
            logger.error(f"LLM API call failed: {e}")
            return record.parsed_result

    # ── Async batch methods ─────────────────────────────────────

    async def _async_call_single(self, semaphore: asyncio.Semaphore,
                                  system_prompt: str, user_prompt: str,
                                  call_type: str, archetype_key: str,
                                  tract_id: str) -> dict:
        """Single async LLM call with semaphore-controlled concurrency."""
        async with semaphore:
            self.call_count += 1
            record = LLMCallRecord(
                call_id=self.call_count,
                call_type=call_type,
                archetype_key=archetype_key,
                tract_id=tract_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            start_time = time.time()

            try:
                response = await self.async_client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                record.latency_ms = (time.time() - start_time) * 1000
                content = response.choices[0].message.content.strip()
                record.raw_response = content

                if response.usage:
                    record.input_tokens = response.usage.prompt_tokens
                    record.output_tokens = response.usage.completion_tokens
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens

                # Parse JSON
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                parsed = json.loads(content)
                record.parsed_result = parsed
                self.call_log.append(record)
                return parsed

            except json.JSONDecodeError as e:
                record.latency_ms = (time.time() - start_time) * 1000
                record.is_fallback = True
                record.fallback_reason = f"JSON parse error: {e}"
                fallback = {"satisfaction": 5.0, "would_move": False, "key_reasons": ["parse_error"]}
                record.parsed_result = fallback
                self.call_log.append(record)
                logger.warning(f"LLM returned invalid JSON: {content[:200]}... Error: {e}")
                return fallback
            except Exception as e:
                record.latency_ms = (time.time() - start_time) * 1000
                record.is_fallback = True
                record.fallback_reason = f"API error: {e}"
                record.error = str(e)
                fallback = {"satisfaction": 5.0, "would_move": False, "key_reasons": ["api_error"]}
                record.parsed_result = fallback
                self.call_log.append(record)
                logger.error(f"Async LLM call failed: {e}")
                return fallback

    def call_batch(self, call_specs: list[dict]) -> list[dict]:
        """
        Execute multiple LLM calls concurrently using asyncio.

        Each call_spec is a dict with keys:
            system_prompt, user_prompt, call_type, archetype_key, tract_id

        Returns list of parsed results in the same order as call_specs.
        Provides ~5-10x speedup over sequential calls.
        """
        if not call_specs:
            return []

        # If only 1 call, just do it synchronously
        if len(call_specs) == 1:
            spec = call_specs[0]
            return [self._call(
                spec["system_prompt"], spec["user_prompt"],
                call_type=spec.get("call_type", ""),
                archetype_key=spec.get("archetype_key", ""),
                tract_id=spec.get("tract_id", ""),
            )]

        return asyncio.run(self._run_batch(call_specs))

    async def _run_batch(self, call_specs: list[dict]) -> list[dict]:
        """Run a batch of calls concurrently with semaphore."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        tasks = [
            self._async_call_single(
                semaphore,
                spec["system_prompt"], spec["user_prompt"],
                call_type=spec.get("call_type", ""),
                archetype_key=spec.get("archetype_key", ""),
                tract_id=spec.get("tract_id", ""),
            )
            for spec in call_specs
        ]
        return await asyncio.gather(*tasks)

    # ── Convenience batch methods ───────────────────────────────

    def batch_assess_satisfaction(self, tasks: list[dict]) -> list[dict]:
        """
        Batch satisfaction assessment.

        Each task: {archetype_desc, neighborhood_desc, archetype_key, tract_id}
        Returns list of parsed results in same order.
        """
        call_specs = []
        for t in tasks:
            user_msg = (
                f"Household profile:\n{t['archetype_desc']}\n\n"
                f"Current neighborhood conditions:\n{t['neighborhood_desc']}\n\n"
                f"Evaluate this household's satisfaction with their current neighborhood."
            )
            call_specs.append({
                "system_prompt": SATISFACTION_SYSTEM_PROMPT,
                "user_prompt": user_msg,
                "call_type": "satisfaction",
                "archetype_key": t.get("archetype_key", ""),
                "tract_id": t.get("tract_id", ""),
            })
        return self.call_batch(call_specs)

    def batch_evaluate_move(self, tasks: list[dict]) -> list[dict]:
        """
        Batch move evaluation.

        Each task: {archetype_desc, current_desc, candidates_desc, archetype_key, tract_id}
        Returns list of parsed results in same order.
        """
        call_specs = []
        for t in tasks:
            user_msg = (
                f"Household profile:\n{t['archetype_desc']}\n\n"
                f"Current neighborhood:\n{t['current_desc']}\n\n"
                f"Candidate neighborhoods:\n{t['candidates_desc']}\n\n"
                f"Pick the best neighborhood to move to, or recommend staying."
            )
            call_specs.append({
                "system_prompt": MOVE_EVAL_SYSTEM_PROMPT,
                "user_prompt": user_msg,
                "call_type": "move_eval",
                "archetype_key": t.get("archetype_key", ""),
                "tract_id": t.get("tract_id", ""),
            })
        return self.call_batch(call_specs)

    # ── Query methods ───────────────────────────────────────────

    def get_call_log_since(self, start_id: int) -> list[LLMCallRecord]:
        """Return call records with call_id > start_id."""
        return [r for r in self.call_log if r.call_id > start_id]

    def get_current_call_id(self) -> int:
        """Return the current max call_id (for tracking per-step calls)."""
        return self.call_count

    def get_stats(self) -> dict:
        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
