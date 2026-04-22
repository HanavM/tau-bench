As part of the University of California, San Francisco [Abbasi Laboratory]((https://abbasilab.org/)), Hanav Modasiya (intern), Dr. Patrick Xian, and Dr. Reza Abbasi-Asl developed a framework for implementing stochastic behavioral interventions on multi-turn LLM agents.

The paper is currently under review, but the premise is that we developed an agent that recursively determines N failure points in a multi-turn LLM agent's reasoning trajectory and plans N behavioral interventions to combat those failure points. 

This is the implementation of stochastic behavioral interventions on the [τ-bench]([url](https://taubench.com/#home)) benchmark, specifically its airline and retail categories. The environment is forked from the the τ-bench repository, and the primary code for behavioral interventions are in /tau-bench/agents/chat_react_agent_intervened.py, /tau-bench/agents/intervening-prompts.py, and /tau-bench/run.py. 

Through our evaluations, the following results were achieved:

### Pass Rate before and after behavioral interventions

| Worker / User Agent | Intervenor      | τ-bench Airline  | τ-bench Retail |
|--------------------|------------------|------------------|----------------|
| GPT-4o-mini        | None             | 20.0             | 17.4           |
| GPT-4o-mini        | GPT-4o-mini (N=5)| 50.0 (+150%)     | —              |
| GPT-4o-mini        | GPT-5-mini (N=5) | 52.0 (+160%)     | 44.4 (+155%)   |
|--------------------|------------------|------------------|----------------|
| GPT-4.1-mini       | None             | 42.0             | 35.7           |
| GPT-4.1-mini       | GPT-5-mini (N=5) | 70.0 (+66.7%)    | 71.3 (+99.7%)  |
|--------------------|------------------|------------------|----------------|
| GPT-5-mini         | None             | 31.0             | 50.4           |
| GPT-5-mini         | GPT-4o-mini (N=5)| 60.0 (+93.5%)    | —              |
| GPT-5-mini         | GPT-5-mini (N=5) | 54.0 (+74.2%)    | 81.7 (+62.1%)  |
