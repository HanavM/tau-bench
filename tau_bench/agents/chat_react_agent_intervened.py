# Copyright Sierra

import json
from litellm import completion
from docent import Docent

from openai import OpenAI
import openai
import os 
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)


#docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import parse_chat_message
from docent.samples import get_inspect_fpath
from pydantic_core import to_jsonable_python
from typing import Any
from docent.data_models import BaseAgentRunMetadata
from pydantic import Field
from docent.data_models.chat import SystemMessage, UserMessage, AssistantMessage, ToolMessage, ContentReasoning
import re
from tau_bench.agents.intervening_prompts import questioning_agent_prompt, SEARCH_PROMPT, SINGLE_RUN_CITE_INSTRUCTION, questioning_agent_prompt_working_backwards
import pprint
from docent.data_models.chat import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ContentText,
    ContentReasoning,
    ToolCall,
)

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import EnvRunResult, RunConfig
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)
from typing import Optional, List, Dict, Any, Tuple

class CustomTauAgentRunMetadata(BaseAgentRunMetadata):
    task_id: str = Field(
        description="The ID of the 'benchmark' or 'set of evals' that the transcript belongs to"
    )

    sample_id: str = Field(
        description="The specific task inside of the `task_id` benchmark that the transcript was run on"
    )
    epoch_id: int = Field(
        description="Each `sample_id` should be run multiple times due to stochasticity; `epoch_id` is the integer index of a specific run."
    )

    model: str = Field(description="The model that was used to generate the transcript")

    scoring_metadata: dict[str, Any] | None = Field(
        description="Additional metadata about the scoring process"
    )

    additional_metadata: dict[str, Any] | None = Field(
        description="Additional metadata about the transcript"
    )



class ChatReActAgentIntervened(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
    ) -> None:
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )
        message = res.choices[0].message
        action_str = message.content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            # this is a hack
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        return message.model_dump(), action, res._hidden_params["response_cost"]

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]

        total_cost = 0.0
        info = {}
        for _ in range(max_num_steps):
            message, action, cost = self.generate_next_step(messages)
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
            total_cost += cost
            if response.done:
                break
        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )

    def add_intervention(trajectory, intervention_text, intervention_id):
        idx_b = -1 if (intervention_id.find("B") == -1) else intervention_id.find("B")
        idx_intervention = int(intervention_id[idx_b+1:])

        new_trajectory = trajectory[:idx_intervention+1]
        new_trajectory.append(
            {
                "role":"system",
                "content": "[*INTERVENTION*: " +intervention_text + "]"
            }
        )

        return new_trajectory

    def run_intervention(
            self, N, result: EnvRunResult, env: Env, task_index: Optional[int] = None
    ):
        
        def load_TAU_Reasoning_inspect_log(log) -> list[AgentRun]:
            agent_runs: list[AgentRun] = []
            for sample in log:
                scores: dict[str, int | float | bool] = {}
                scores["correct"] =  (sample["reward"] == 1.0)

                metadata = CustomTauAgentRunMetadata(
                        task_id="airline",
                        sample_id=str(sample["task_id"]),
                        epoch_id=int(sample["trial"]),
                        model=self.model,
                        scores=scores,
                        additional_metadata=None,
                        scoring_metadata=None,
                    )
                

                messages = []

                for idx, message in enumerate(sample["traj"]):
                    if message["role"] == "tool":
                        messages.append(ToolMessage(id = str(idx),content=message["content"], tool_call_id=message["tool_call_id"], function=message["name"]))

                    elif message["role"] == "assistant":
                        contentstr = message["content"]
                        if contentstr == None:
                            contentstr = ""
                        if message["tool_calls"]:
                            contentstr += json.dumps(message["tool_calls"], indent=2)
                        if message["function_call"]:
                            contentstr += str(message["function_call"])
                        if message["annotations"]:
                            contentstr += message["annotations"]

                        messages.append(AssistantMessage(id = str(idx),content=contentstr))

                    else:
                        messages.append(UserMessage(id = str(idx),content=message["content"]))


                agent_runs.append(
                AgentRun(
                    transcripts={
                        "default": Transcript(
                            messages=messages
                        )
                    },
                    metadata=metadata,
                )
                )

            return agent_runs

        def execute_search(text, query, model):

            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user","content":SEARCH_PROMPT.format(item=text, search_query=query, SINGLE_RUN_CITE_INSTRUCTION=SINGLE_RUN_CITE_INSTRUCTION)}],
                max_tokens=4096,
                temperature = 0.1
            )



            return response.choices[0].message.content.strip()

        def load_TAU_reAct_extra_data(transcript):
            return transcript["traj"][0], transcript["info"]["reward_info"], transcript["info"]["task"]

        def add_intervention(trajectory, intervention_text, intervention_id):
            idx_b = -1 if (intervention_id.find("B") == -1) else intervention_id.find("B")
            idx_intervention = int(intervention_id[idx_b+1:])

            new_trajectory = trajectory[:idx_intervention+1]
            new_trajectory.append(
                {
                    "role":"system",
                    "content": "[*INTERVENTION*: " +intervention_text + "]"
                }
            )

            return new_trajectory



        transcript = result.model_dump()
        rubric, metadata, user_task = load_TAU_reAct_extra_data(transcript)
        agent_run_docent = load_TAU_Reasoning_inspect_log([transcript])

        print("N:", N)


        conversation_history = []
        conversation_history.append({
            "role": "system",
            "content": questioning_agent_prompt_working_backwards.format(rubric=rubric, user_task=user_task,metadata=metadata, N=N),
        })   
        
        turns = 0

        while turns < 30:
            turns+=1
            response = openai.chat.completions.create(
                model=self.model,
                messages=conversation_history,
                max_tokens=4096,
                temperature=0.1
            )

            reply = response.choices[0].message.content.strip()
            match = re.search(r'<query>(.*?)</query>', reply, re.DOTALL)

            conversation_history.append({
                "role": "assistant",
                "content": reply,
            })



            if match:
                query_text = match.group(1).strip()
                conversation_history[-1]["tool_calls"] = [{"function": { "arguments": query_text, "name": "querying_tool"      }, "id": "12345","type": "function"}]
                tool_response = execute_search(agent_run_docent[0].transcripts["default"].to_str(), query_text, self.model).strip()

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": "12345",
                    "content": tool_response
                })
                continue



            match = re.search(r'<answer>(.*?)</answer>', reply, re.DOTALL)

            if match:
                answer_text = match.group(1).strip()
                try:
                    

                    answer_list = json.loads(answer_text)
                    # print("answer:", answer_list)

                    possible_new_trajectories = []
                    for intervention in answer_list:
                        intervention_text = intervention["intervention_text"]
                        intervention_id = intervention["id"]
                        
                        possible_new_trajectories.append(add_intervention(transcript["traj"], intervention_text, intervention_id))

                
                    # pprint.pprint(conversation_history)
                    return answer_list, conversation_history

                except json.JSONDecodeError:
                    print("Error decoding JSON.")
                break


            else:
                print("model did not call query tool or generate intervention")
                break
        

        print("no changes with intervention, error must have happened")
        return [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": response.observation},
            ], conversation_history
        



    def solve_with_intervention(
        self, messages, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
    
    
        total_cost = 0.0
        info = {}
        for _ in range(max_num_steps):
            message, action, cost = self.generate_next_step(messages)
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
            total_cost += cost
            if response.done:
                break
        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )
    



REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy.
"""


ACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You should not use made-up or placeholder arguments.

The Action will be parsed, so it must be valid JSON.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}
```

Your response can be like this:
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
"""
