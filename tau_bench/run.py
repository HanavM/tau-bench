# Copyright Sierra

import os
import json
import random
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot", "react-intervened"], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"
    
    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_model}-{config.user_strategy}_{time_str}"

    ckpt_path_intervened = ckpt_path + "_intervened"
    agent_conv_history_path = ckpt_path + 'intervening_agent_conversation_history.json'
    ckpt_path += ".json"
    ckpt_path_intervened += ".json"



    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )
    end_index = (
        len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    results_intervened: List[EnvRunResult] = []
    total = []
    improved_count = []
    intervened_succesful_count = []
    agent_conversation_histories = []
    lock = multiprocessing.Lock()
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        # if config.shuffle:
        #     random.shuffle(idxs)

        print(f"N from config {config.best_of_N}")

        def _run(idx: int, N=config.best_of_N) -> EnvRunResult:
        
            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
            )

            print(f"Running task {idx}")
            try:
                res = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                )
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                # result.info,
            )
            print("-----")

            #implement intervention
            if (config.run_intervention):
                print("Starting intervention on task id=", idx)
                try:
                    
                    #get intervention possibilites and conv history of intervening agent
                    answer_list, intervening_agent_conversation = agent.run_intervention(
                        env=isolated_env,
                        task_index=idx,
                        result=result,
                        N=N
                    )

                    #add conv history to all conv histories
                    agent_conversation_histories.append({"task_id":idx,"traj":intervening_agent_conversation})

                    transcript = result.model_dump()
                    possible_new_trajectories = []
                    

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
                    
                    does_improve = False
                    passed_intervened = False
                    best_intervened_score = 0
                    #loop through all intervention possibilites
                    for best_of_n_iterator, intervention in enumerate(answer_list):
                        print(f"trying out task id={idx}, intervention {best_of_n_iterator}")
                        
                        intervention_text = intervention["intervention_text"]
                        intervention_id = intervention["id"]
                        print(f"intervention id: {intervention_id} intervention txt: {intervention_text}")
                        
                        #add intervention to trajectory
                        new_intervened_trajectory = add_intervention(transcript["traj"], intervention_text, intervention_id)

                        possible_new_trajectories.append(new_intervened_trajectory)

                        
                        #Run again but with intervention
                        res_intervened = agent.solve_with_intervention(
                            env=isolated_env,
                            task_index=idx,
                            messages=new_intervened_trajectory
                        )
                        

                        # print("ran new agent task with intervened transcript")



                        #compile result of intervention
                        result_intervened = EnvRunResult(
                            intervened_message = intervention_text,
                            intervened_index = intervention_id,
                            improved = (result.reward == 0 and res_intervened.reward != 0),
                            task_id=idx,
                            reward=res_intervened.reward,
                            info=res_intervened.info,
                            traj=res_intervened.messages,
                            trial=i,
                        )
                        
                        if result.reward == 0 and result_intervened.reward != 0:
                            print(f"***IMPROVED*** task_id={idx} at location={intervention_id}")
                            does_improve = True
                        
                        if result_intervened.reward != 0:
                            passed_intervened = True
                            best_intervened_score = max(result_intervened.reward, best_intervened_score)

                        #save result of intervention
                        with lock:
                            data = []
                            if os.path.exists(ckpt_path_intervened):
                                with open(ckpt_path_intervened, "r") as f:
                                    data = json.load(f)
                            with open(ckpt_path_intervened, "w") as f:
                                json.dump(data + [result_intervened.model_dump()], f, indent=2)

                    total.append(True)
                    if (does_improve == True):
                        improved_count.append(True)
                    # if (passed_intervened == True):
                    intervened_succesful_count.append(best_intervened_score)



                except Exception as e:
                    result_intervened = EnvRunResult(
                        task_id=idx,
                        reward=0.0,
                        info={"error": str(e), "traceback": traceback.format_exc()},
                        traj=[],
                        trial=i,
                    )
                
                print(
                    "✅" if result_intervened.reward == 1 else "❌",
                    f"task_id={idx}",
                    # result_intervened.info,
                )



            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)

            results_intervened.extend
            
            return result


        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = executor.map(_run, idxs)
            res = list(res)
            results.extend(res)

    display_metrics(results)

    if (config.run_intervention):
        average = sum(intervened_succesful_count) / len(total)
        print("Average reward (intervened):", average)
        print("total samples:", len(total))
        print("# of samples with intervention with score > 0:", len(intervened_succesful_count))
        print("# of samples improved from a score of 0 with intervention:", len(improved_count))
        # print("percent of samples passed with intervention:", len(intervened_succesful_count) / len(total))
        print("percent of missed samples improved with intervention:", len(improved_count) / len(total))
        print("scores for interventions:", intervened_succesful_count)


        with open(agent_conv_history_path, 'w') as json_file:
            json.dump(agent_conversation_histories, json_file, indent=4)
            print(f"Agent conversation history successfully written to '{agent_conv_history_path}'")




    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")

    return results


def agent_factory(
    tools_info: List[Dict[str, Any]], wiki, config: RunConfig
) -> Agent:
    if config.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    
    elif config.agent_strategy == "react-intervened":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent_intervened import ChatReActAgentIntervened

        return ChatReActAgentIntervened(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    
    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
        assert config.few_shot_displays_path is not None, "Few shot displays path is required for few-shot agent strategy"
        with open(config.few_shot_displays_path, "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
