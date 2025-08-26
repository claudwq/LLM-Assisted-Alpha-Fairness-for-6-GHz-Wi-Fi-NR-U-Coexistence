# LLM-Assisted-Alpha-Fairness-for-6-GHz-Wi-Fi-NR-U-Coexistence
This repository contains the official implementation for the paper: "LLM-Assisted Alpha-Fairness for 6 GHz Wi-Fi/NR-U Coexistence: An Agentic Orchestrator for Throughput, Energy, and SLA."

This project introduces and simulates an agentic controller that uses a Large Language Model (LLM) to manage spectrum sharing between Wi-Fi and 5G NR-U in the unlicensed 6 GHz band. The agent intelligently balances the competing objectives of network throughput, energy efficiency, and Service Level Agreement (SLA) satisfaction by separating high-level policy reasoning from low-level execution.

🚀 Key Features
Hybrid Architecture: Combines an LLM for high-level, interpretable policy suggestions with a deterministic optimizer for low-level, safe resource allocation.

Multi-Objective Optimization: Dynamically balances throughput, energy consumption, and user SLA deadlines.

Alpha-Fairness Control: Leverages the 
alpha-fairness utility function to provide a flexible spectrum allocation policy.

Two Modes of Operation:

Rule-Based: A deterministic scheduler based on fixed rules.

LLM-Assisted: An advanced scheduler where an LLM dynamically sets policy parameters each epoch.

Detailed Simulation: Models a multi-user, multi-channel environment with coexisting Wi-Fi and NR-U traffic.

Reproducibility: Outputs detailed CSV logs and plots to easily reproduce the paper's findings.

🏛️ System Architecture
The controller operates on a two-layer agentic architecture:

Policy Orchestrator (LLM): At the start of each scheduling epoch, the agent constructs a detailed JSON prompt summarizing the network state (channel contention, user backlogs, CQI, battery levels, SLA deadlines, etc.). This prompt is sent to an LLM (e.g., GPT-4o), which analyzes the complex state and proposes a set of simple, interpretable policy "knobs." These knobs include the fairness index (
alpha), per-technology duty-cycle caps, and priority weights.

Deterministic Executor (Optimizer): The knobs suggested by the LLM are passed to a deterministic optimizer. This layer validates the policy, clamps any unsafe values, and computes the final resource allocation using an 
alpha-fair utility function. This ensures that the final actions are always safe, predictable, and compliant with system constraints.

This separation of concerns allows the system to leverage the powerful reasoning capabilities of LLMs without sacrificing the safety and auditability required for network management.

🏁 Getting Started
Prerequisites
Python 3.9 or higher


Set up API Key (for LLM mode):
To run the LLM-assisted agent, you must set your OpenAI API key as an environment variable.

export OPENAI_API_KEY='your-api-key-here'

🖥️ Usage
The main script is 2_fixed_spectrum_agent.py. You can run it in two primary modes.

1. Rule-Based Policy (Default)
This mode runs the simulation with a fixed, rule-based scheduler without calling an LLM.

python 2_fixed_spectrum_agent.py

2. LLM-Assisted Policy
This mode enables the LLM to dynamically set the policy for each epoch.

python 2_fixed_spectrum_agent.py --use-llm --llm-model gpt-4o

Command-Line Arguments
--use-llm: (Flag) Enable the LLM-based policy orchestrator.

--llm-model: (str) The model to use for the LLM agent (e.g., gpt-4o, gpt-3.5-turbo). Defaults to gpt-4o.

--no-plots: (Flag) Suppress the generation of matplotlib plots and only output the result CSV files.

--outdir: (str) The directory to save output CSV files. Defaults to ./output.

📈 Outputs
The script will generate the following files in the specified output directory (./output/ by default):

allocations_by_alpha.csv: Detailed resource allocation results for each evaluated alpha value.

user_goodputs_by_alpha.csv: Per-user throughput metrics for each alpha value.

If plotting is enabled, interactive charts showing the alpha-fairness utility and per-user goodput will be displayed after the simulation completes.



📜 License
