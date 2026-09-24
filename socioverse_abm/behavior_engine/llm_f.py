"""
Basic LLM helpers shared by the tasks' LLM behavior functions.

The OpenAI-compatible client is created lazily on first use from OPENAI_API_KEY and
(optionally) OPENAI_BASE_URL; when OPENAI_BASE_URL is unset the official OpenAI
endpoint is used. Importing this module never requires a key.
"""

import os
import re
import json
from socioverse_abm.behavior_engine.prompt import UNIVERSAL_PROMPT

_client = None


def get_client():
    """Return the shared OpenAI client, creating it on first use."""
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
    return _client


def __getattr__(name):
    # Backward compatibility: `llm_f.client` still resolves (lazily).
    if name == "client":
        return get_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate(model, prompt, sys_prompt="Your are a helpful assistant", max_tokens=2048, temperature=0.7, **kwargs):
    if "gpt-5" in model:
        response = get_client().responses.create(
            model=model,
            max_output_tokens=max_tokens,
            temperature=temperature,
            input=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"}
        )
        # print(response)
        # print("#"*20)
        return response.output_text
    
    else:
        # if model.startswith("qwen3-235b-a22b"):
        #     if "thinking" in model:
        #         prompt = f"{prompt}/think"
        #         model = "qwen3-235b-a22b"
        #     else:
        #         prompt = f"{prompt}/no_think"

        response = get_client().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            **kwargs
            )
        # print(response)
        # print("#"*20)
        return response.choices[0].message.content


def extract_json_from_gpt(content, model):
    # print(content)
    if model == 'deepseek-r1':
        content = content.split('</think>')[1]
        content = content.strip()
    if '```json' in content:
        json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parsing failed after regex extraction: {e}")
        else:
            raise ValueError("No JSON content found in the response")
    else:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(content)
            raise ValueError(f"JSON parsing failed: {e}")


def generate_and_parser(model, prompt, sys_prompt="Your are a helpful assistant", max_tokens=2048, temperature=0.7):
    max_times = 3
    cunt_time = 0
    while True:
        try:
            response = generate(model, prompt, sys_prompt, max_tokens, temperature)
            # print(response)
            format_response = extract_json_from_gpt(response, model)
            # print(format_response)
            # waitkey = input("Press Enter to continue...")
            return format_response
        except Exception as e:
            cunt_time += 1
            if cunt_time == max_times:
                raise Exception(e)


def get_embedding(model, texts, embed_dim=3072):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, list):
        texts = texts
    else:
        raise TypeError("`texts` must be a string or a list of strings")
    
    response = get_client().embeddings.create(
        model=model,
        input=texts,
        dimensions=embed_dim
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings


def parse_attributes(scenario, attributes):
    if scenario == "sugarscape":
        # prompt = f"You are a citizen in a sugarscape environment. Your current sugar level is {attributes['sugar']} and your metabolism rate is {attributes['metabolism']}. You can see the following locations within your vision range:\n"
        # for loc, sugar in attributes["candidates"]:
        #     prompt += f"Location {loc} has {sugar} sugar.\n"
        # prompt += "Based on this information, which location should you move to in order to maximize your sugar intake while considering your metabolism rate? Please provide the coordinates of the location you choose to move to.\n Format your answer as {\"move\": \"(x, y)\"} without any other output."

        task_desc = "You are a citizen in a sugarscape environment. Your goal is to maximize your sugar intake while managing your metabolism rate."
        attribute_desc = "Your current sugar level is {attributes['sugar']} and your metabolism rate is {attributes['metabolism']}."
        env_configs = "You can see the following locations within your vision range:\n"
        for loc, sugar in attributes["candidates"]:
            env_configs += f"Location {loc} has {sugar} sugar.\n"
        rule_desc = "Based on this information, which location should you move to in order to maximize your sugar intake while considering your metabolism rate? Please provide the coordinates of the location you choose to move to."
        res_template = "Format your answer as {\"move\": \"(x, y)\"} without any other output, where x, y are integers representing the coordinates."
    
    if scenario == "HK_model":
        # prompt = f"You are an agent with your opinion in range [0, 1). Your current opinion is {attributes['self_opinion']}. You have the following neighbor opinions within your tolerance range of {attributes['epsilon']}:\n\n"
        # prompt += f"{attributes['neighbor_opinions']}\n\n"
        # prompt += "Based on these opinions, please update your opinion. Format your answer as {\"new_opinion\": updated_opinion} without any other output. The updated opinion should also be a float number in the range [0, 1)."

        task_desc = "You are an agent with your opinion in range [0, 1). Your goal is to update your opinion based on the opinions of your neighbors within your tolerance range."
        attribute_desc = f"Your current opinion is {attributes['self_opinion']}."
        env_configs = f"You have the following neighbor opinions within your tolerance range of {attributes['epsilon']}:\n{attributes['neighbor_opinions']}"
        rule_desc = "Based on these opinions, please update your opinion."
        res_template = "Format your answer as {\"new_opinion\": updated_opinion} without any other output. The updated opinion should also be a float number in the range [0, 1)."
        
    if scenario == "SIR_model":
#         task_desc = "You are simulating the spread of a rumor in a social network. Each node in the network can be in one of three states: Susceptible (0), Spreader (1), or Removed (2)."
#         attribute_desc = f"""
# - A Susceptible node (0) can become a Spreader (1) with a probability of {attributes['beta']} if it has at least one neighboring Spreader.
# - A Spreader (1) can become Removed (2) with a probability of {attributes['gamma']}.
# - A Removed node (2) cannot change its state."""
#         env_configs = f"Now, you have the following information about the current state of the network:\n{attributes['neighbor_list']}"
#         rule_desc = f"""Based on this information and rules of probability, please determine the updated states of the nodes after one iteration.
# First evaluate the states of the Susceptible neighbors ({attributes['beta']}: 0 -> 1), then evaluate the Spreader itself's potential state change ({attributes['gamma']}: 1 -> 2)."""
#         res_template = f"Format your answer as JSON format: {{\"updated_nodes\": [{{\"node_id\": new_state}}, ...]}} without any other output, where new_state is 0, 1, or 2. If no nodes change their states, return {{\"updated_nodes\": []}}.\nNotice: only output the nodes that have changed their states. Do not include nodes that remain in the same state."
        
        task_desc = "You are simulating the spread of a rumor in a social network. Each node in the network can be in one of three states: Susceptible (0), Spreader (1), or Removed (2)."
        if attributes["type"] == "spreader":
            attribute_desc = f"You are a Spreader (1). You can become Removed (2) with a threshold of {attributes['gamma']}, otherwise you remain a Spreader (1)."
            env_configs = f"Now, you are going to evaluate your own potential state change. Your current value is {attributes['current_prob']}. Change your state into Removed (2) if your current value is less than the threshold; otherwise, remain a Spreader (1)."
        if attributes["type"] == "susceptible":
            attribute_desc = f"You are a Susceptible (0). If you have at least one neighboring Spreader, you can become a Spreader (1) with a threshold of {attributes['beta']}, otherwise you remain Susceptible (0)."
            env_configs = f"Now, at least one of your neigbhbor is a Spreader. You are going to evaluate your own potential state change. Your current value is {attributes['current_prob']}. Change your state into Spreader (1) if your current value is less than the threshold; otherwise, remain a Susceptible (0)."
        
        # rule_desc = f"Based on the above information, first generate a RANDOM FLOAT number between 0 and 1, then determine your updated state based on this number. If the generated probability is less than the threshold for state change, you will change your state accordingly; otherwise, you will remain in your current state."
        # res_template = f"Format your answer as JSON format: {{\"probability\": #prob, \"updated_state\": #state}} without any other output, where #prob is your generated random float number between 0 and 1, #state is an integer of 0, 1, or 2."
        rule_desc = f"Based on the above information, determine your updated state."
        res_template = f"Format your answer as JSON format: {{\"probability\": #prob, \"updated_state\": #state}} without any other output, where #prob is the current probability that you received, #state is an integer of 0, 1, or 2."

    if scenario == "Schelling_model":
        # Uses the UNIVERSAL_PROMPT layout
        type_name = f"Type {attributes['agent_type']}"
        similar = attributes['similar_neighbors']
        threshold = attributes['homophily_threshold']
        
        task_desc = f"You are a {type_name} agent in a residential choice simulation. Your goal is to achieve satisfaction by having at least {threshold} similar neighbors in your vicinity."
        
        attribute_desc = f"""
- Position: {attributes['agent_pos']}
- Agent type: {type_name}
- Similar neighbors: {similar} out of {attributes['total_neighbors']} total neighbors
- Preference threshold: at least {threshold} similar neighbors
- Current satisfaction: {'SATISFIED' if similar >= threshold else 'UNSATISFIED'}"""
        
        if similar < threshold:
            env_configs = f"""
You are UNSATISFIED (only {similar}/{threshold} similar neighbors).
Available vacant locations to move to: {attributes.get('sample_empty_cells', [])[:5]}
Grid size: {attributes.get('grid_size', 'N/A')}
Total empty cells: {attributes.get('empty_cells_count', 'N/A')}"""
        else:
            env_configs = f"""
You are SATISFIED ({similar}/{threshold} similar neighbors met).
Grid size: {attributes.get('grid_size', 'N/A')}"""
        
        rule_desc = f"""
Based on your current situation:
- If you are SATISFIED ({similar} >= {threshold}), you should likely stay at your current position.
- If you are UNSATISFIED ({similar} < {threshold}), you should decide whether to move to a vacant location to improve your satisfaction.
You can choose to move to one of the available vacant locations or stay at your current position."""
        
        res_template = """Format your response EXACTLY as JSON:
- To relocate: {"decision": "move", "target_position": [x, y]}
- To stay: {"decision": "stay"}
Output ONLY valid JSON, no other text."""
    
    if scenario == "Civil_Violence_model":
        # Uses the UNIVERSAL_PROMPT layout
        task_desc = "You are simulating an agent in a social dynamics model. You need to decide whether to become active (protest) or remain quiet based on your grievance level, risk tolerance, and environmental factors."
        
        attribute_desc = f"""
- Position: {attributes['citizen_pos']}
- Discontent level (hardship): {attributes['hardship']} (0-1 scale, higher = more discontent)
- Risk tolerance (risk_aversion): {attributes['risk_aversion']} (0-1 scale, higher = more cautious)
- Net grievance: {attributes['grievance']} (computed from hardship and legitimacy)
- Current state: {attributes['current_state']}"""
        
        env_configs = f"""
Environment Factors:
- Authority legitimacy: {attributes['government_legitimacy']} (0-1, higher = more legitimate)
- Estimated intervention probability (arrest_probability): {attributes['arrest_probability']} (0-1)
- Authority agents nearby: {attributes['cops_nearby']}
- Active agents nearby: {attributes['actives_nearby']}
- Activation threshold: {attributes['threshold']}"""
        
        rule_desc = f"""
Decision Logic:
Become ACTIVE if: (Net_Grievance - Risk_Tolerance × Intervention_Probability) > Threshold ({attributes['threshold']})
Otherwise, remain QUIET.

Consider these factors in your decision:
1. Your grievance level versus intervention risk
2. Number of active agents nearby (collective action effect)
3. Number of authority agents nearby (enforcement risk)
4. Your personal risk tolerance"""
        
        res_template = """Format your response EXACTLY as JSON:
- To become active: {"decision": "active"} or {"decision": "protest"}
- To remain quiet: {"decision": "quiet"}
Output ONLY valid JSON, no other text."""

    
    prompt = UNIVERSAL_PROMPT.format(
        task_desc=task_desc,
        attribute_desc=attribute_desc,
        env_configs=env_configs,
        rule_desc=rule_desc,
        res_template=res_template)

    # print(prompt)
    # waitkey = input("Press Enter to continue...")
    return prompt
