#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrum Agent with Alpha-Fairness + (Optional) LLM Policy Orchestration.

Usage:
  # Rule-based policy (no LLM)
  python llm_spectrum_agent_fairness.py

  # Enable LLM policy selection (requires OPENAI_API_KEY)
  python llm_spectrum_agent_fairness.py --use-llm --llm-model gpt-4o

  # Export CSV only, no plots
  python llm_spectrum_agent_fairness.py --no-plots
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, replace  # Added 'replace' import
from typing import Dict, List, Tuple
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Conditional OpenAI import - moved inside functions to avoid import errors
# when OPENAI_API_KEY is not set and not using LLM

# -------------------------
# Models & helpers
# -------------------------

SE_TABLE = {
    0: 0.0, 1: 0.1523, 2: 0.2344, 3: 0.3770, 4: 0.6016,
    5: 0.8770, 6: 1.1758, 7: 1.4766, 8: 1.9141, 9: 2.4063,
    10: 2.7305, 11: 3.3223, 12: 3.9023, 13: 4.5234, 14: 5.1152, 15: 5.5547
}
PRIORITY_ALPHA = {"emergency": 5.0, "high": 3.0, "normal": 1.5, "bulk": 1.0}
POWER_PROFILES = {"low": (0.25, 0.9), "med": (0.5, 1.0), "high": (0.9, 1.08)}

def beta_from_battery(B: float) -> float:
    if B < 0.20: return 3.0
    if B < 0.40: return 2.0
    return 1.0

def gamma_from_latency_ms(D: float) -> float:
    if D <= 20: return 4.0
    if D <= 50: return 2.5
    return 1.5

def latency_required_rate(bits: float, D_ms: float) -> float:
    if D_ms <= 0: return float('inf')
    return bits / (D_ms / 1000.0)

def lbt_loss_fraction(own_duty: float, sensed_busy: float, base_fail: float) -> float:
    add = 0.6 * max(0.0, own_duty) * max(0.0, sensed_busy) + 0.2 * max(0.0, own_duty + sensed_busy - 1.0)
    return float(min(0.95, max(0.0, base_fail + add)))

@dataclass
class User:
    id: str
    tech: str
    CQI: int
    battery: float
    data_bits: float
    latency_ms: float
    priority: str
    pwr_mode: str

@dataclass
class Channel:
    id: str
    bw_hz: float
    wifi_busy: float
    nru_busy: float
    wifi_lbt_base_fail: float
    nru_lbt_base_fail: float

@dataclass
class Env:
    epoch_s: float
    channels: List[Channel]

@dataclass
class PolicyOut:
    template: str
    duty_caps_wifi: Dict[str, float]
    duty_caps_nru: Dict[str, float]
    alpha: Dict[str, float]
    notes: List[str]
    alpha_fair: float = 1.0  # Optional: LLM-chosen α

def se_eff(u: User, ch: Channel) -> float:
    base = SE_TABLE.get(int(np.clip(u.CQI, 0, 15)), 0.0)
    _, mult = POWER_PROFILES[u.pwr_mode]
    return base * mult

# -------------------------
# Rule-based policy (fallback for LLM)
# -------------------------
def rule_policy_multi(env: Env) -> PolicyOut:
    duty_caps_wifi, duty_caps_nru, notes = {}, {}, []
    for ch in env.channels:
        if ch.wifi_busy > ch.nru_busy + 0.07:
            wifi_cap, nru_cap = 0.65, 0.35
            notes.append(f"{ch.id}: Wi-Fi busier ({ch.wifi_busy:.2f} > {ch.nru_busy:.2f}), favor Wi-Fi.")
        elif ch.nru_busy > ch.wifi_busy + 0.07:
            wifi_cap, nru_cap = 0.35, 0.65
            notes.append(f"{ch.id}: NR-U busier ({ch.nru_busy:.2f} > {ch.wifi_busy:.2f}), favor NR-U.")
        else:
            wifi_cap, nru_cap = 0.50, 0.50
            notes.append(f"{ch.id}: similar loads; split evenly.")
        duty_caps_wifi[ch.id] = max(0.0, min(wifi_cap, 1.0 - ch.wifi_busy * 0.5))
        duty_caps_nru[ch.id]  = max(0.0, min(nru_cap,  1.0 - ch.nru_busy  * 0.5))
    return PolicyOut("latency_first_energy_aware_multi", duty_caps_wifi, duty_caps_nru, 
                     PRIORITY_ALPHA.copy(), notes, alpha_fair=1.0)

# -------------------------
# LLM interface: state summary -> structured policy (JSON Schema)
# -------------------------

def build_state_json(users: List[User], env: Env) -> Dict:
    return {
        "epoch_s": env.epoch_s,
        "channels": [
            {
                "id": ch.id, "bw_hz": ch.bw_hz,
                "wifi_busy": ch.wifi_busy, "nru_busy": ch.nru_busy,
                "wifi_lbt_base": ch.wifi_lbt_base_fail, "nru_lbt_base": ch.nru_lbt_base_fail
            } for ch in env.channels
        ],
        "users": [
            {
                "id": u.id, "tech": u.tech, "CQI": int(u.CQI), "battery": float(u.battery),
                "data_bits": float(u.data_bits), "latency_ms": float(u.latency_ms),
                "priority": u.priority, "pwr_mode": u.pwr_mode
            } for u in users
        ],
        "hints": {"alpha_candidates": [0, 1, 2], "goal": "balanced_latency_energy_fairness"}
    }

def llm_choose_policy(state_json: Dict, model: str = "gpt-4o", timeout: int = 20) -> Dict:
    """
    Call OpenAI Chat Completions API for policy selection.
    Returns dict containing {alpha, caps, priority_weights, ...}.
    """
    try:
        from openai import OpenAI
        client = OpenAI(timeout=timeout)
    except Exception as e:
        raise RuntimeError("OpenAI Python SDK not installed. Please run `pip install openai`.") from e

    # Simple approach without structured output to avoid schema issues
    system_msg = (
        "You are a spectrum policy orchestrator. Return ONLY valid JSON with no additional text. "
        "Given the state, choose: alpha (0, 1, or 2), caps (per-channel duty caps for wifi/nru as decimals 0-1), "
        "and priority_weights (emergency, high, normal, bulk as numbers 0.1-10)."
    )
    
    user_msg = f"""Given this spectrum state, return a JSON policy with this exact structure:
{{
    "alpha": 0 or 1 or 2,
    "caps": {{
        "ch37": {{"wifi": 0.0-1.0, "nru": 0.0-1.0}},
        "ch39": {{"wifi": 0.0-1.0, "nru": 0.0-1.0}}
    }},
    "priority_weights": {{
        "emergency": 0.1-10.0,
        "high": 0.1-10.0,
        "normal": 0.1-10.0,
        "bulk": 0.1-10.0
    }},
    "constraints": [],
    "rationales": ["reason 1", "reason 2"],
    "fallback": "use_conservative_default"
}}

State: {json.dumps(state_json, indent=2)}

Return ONLY the JSON response, no other text."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=1, #0.7,
        response_format={"type": "json_object"}  # Basic JSON mode, not structured
    )

    # Parse the response
    try:
        content = resp.choices[0].message.content
        # Remove any markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        policy_obj = json.loads(content.strip())
    except Exception as e:
        raise RuntimeError(f"Failed to parse LLM response as JSON: {e}\nContent: {content}")

    return policy_obj

def coerce_policy_from_llm(llm_obj: Dict, env: Env) -> PolicyOut:
    """
    Safely coerce LLM JSON to PolicyOut with boundary checks and fallbacks.
    """
    # 1) alpha_fair
    alpha_fair = float(llm_obj.get("alpha", 1.0))
    if alpha_fair not in (0.0, 1.0, 2.0):
        alpha_fair = 1.0

    # 2) caps
    caps = llm_obj.get("caps", {})
    duty_caps_wifi, duty_caps_nru = {}, {}
    for ch in env.channels:
        c = caps.get(ch.id, {})
        w = float(c.get("wifi", 0.5))
        n = float(c.get("nru", 0.5))
        # Clamp to [0,1] and consider external busy headroom
        w = max(0.0, min(w, 1.0 - ch.wifi_busy * 0.5))
        n = max(0.0, min(n, 1.0 - ch.nru_busy * 0.5))
        duty_caps_wifi[ch.id] = w
        duty_caps_nru[ch.id] = n

    # 3) priority weights
    pw = llm_obj.get("priority_weights", {})
    alpha_map = {
        "emergency": float(pw.get("emergency", PRIORITY_ALPHA["emergency"])),
        "high":      float(pw.get("high",      PRIORITY_ALPHA["high"])),
        "normal":    float(pw.get("normal",    PRIORITY_ALPHA["normal"])),
        "bulk":      float(pw.get("bulk",      PRIORITY_ALPHA["bulk"]))
    }
    # Reasonable range
    for k in list(alpha_map.keys()):
        alpha_map[k] = float(np.clip(alpha_map[k], 0.1, 10.0))

    notes = []
    for s in llm_obj.get("rationales", [])[:6]:
        if isinstance(s, str) and s.strip():
            notes.append(s.strip())

    return PolicyOut("llm_orchestrated", duty_caps_wifi, duty_caps_nru, 
                     alpha_map, notes, alpha_fair=alpha_fair)

# -------------------------
# Optimizer with Alpha-Fairness
# -------------------------

def optimize_multi_alpha(users: List[User], env: Env, policy: PolicyOut, alpha_fair: float
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    # Stage 1: channel assignment by utility density (including LBT/energy)
    scores = {}
    for u in users:
        a = policy.alpha.get(u.priority, 1.0)
        b = beta_from_battery(u.battery)
        for ch in env.channels:
            eff = se_eff(u, ch)
            rate_per_tau = eff * ch.bw_hz
            tau_probe = 0.05
            base_fail = ch.wifi_lbt_base_fail if u.tech == "wifi" else ch.nru_lbt_base_fail
            other_busy = ch.wifi_busy if u.tech == "wifi" else ch.nru_busy
            loss = lbt_loss_fraction(tau_probe, other_busy, base_fail)
            goodput = rate_per_tau * tau_probe * (1.0 - loss)
            power_w, _ = POWER_PROFILES[u.pwr_mode]
            e_per_bit = power_w / max(1e-6, eff)
            e = e_per_bit * goodput * env.epoch_s
            w_lat = 1.0 + (1.0 if u.latency_ms <= 50 else 0.2)
            util_density = a * (goodput / 1e6) * w_lat / max(1e-3, tau_probe) - b * e / max(1e-3, tau_probe)
            scores[(u.id, ch.id)] = util_density
    assign = {u.id: max(env.channels, key=lambda ch: scores[(u.id, ch.id)]).id for u in users}

    # Stage 2: per-channel duty allocation with alpha-fairness
    rows = []
    per_user_served = {u.id: 0.0 for u in users}

    for ch in env.channels:
        wifi_users = [u for u in users if assign[u.id] == ch.id and u.tech == "wifi"]
        nru_users  = [u for u in users if assign[u.id] == ch.id and u.tech == "nru"]
        cap_wifi   = policy.duty_caps_wifi[ch.id]
        cap_nru    = policy.duty_caps_nru[ch.id]

        tau = {u.id: 0.0 for u in wifi_users + nru_users}
        remain = {"wifi": cap_wifi, "nru": cap_nru}

        def eff_rate(u: User) -> float:
            return se_eff(u, ch) * ch.bw_hz

        # A) Urgent minimum guarantees
        urgent, normal = [], []
        for u in wifi_users + nru_users:
            a = policy.alpha.get(u.priority, 1.0)
            b = beta_from_battery(u.battery)
            score = (1.0 + (1.0 if u.latency_ms <= 50 else 0.2)) * a / (0.5 + b)
            (urgent if (u.latency_ms <= 50 or u.priority in ("emergency", "high")) else normal).append((u, score))
        urgent.sort(key=lambda x: -x[1])
        normal.sort(key=lambda x: -x[1])

        def grant_min(u: User, key: str):
            r_req = min(latency_required_rate(u.data_bits, max(1.0, u.latency_ms)),
                        u.data_bits / max(1e-3, env.epoch_s))
            r_user = eff_rate(u)
            if r_user <= 0: return
            give = min(remain[key], max(0.0, r_req / r_user))
            tau[u.id] += give
            remain[key] -= give

        for u, _ in urgent:
            key = "wifi" if u.tech == "wifi" else "nru"
            if remain[key] > 1e-6:
                grant_min(u, key)

        # B) α-fairness proportional allocation
        eps = 1e-6

        def base_weight(u: User) -> float:
            a = policy.alpha.get(u.priority, 1.0)
            b = beta_from_battery(u.battery)
            return a / (0.5 + b)

        def fair_weight(u: User) -> float:
            served = per_user_served[u.id] + eps
            return base_weight(u) * (served ** (-alpha_fair))

        wifi_pw = [(u, max(1e-9, fair_weight(u))) for u, _ in (urgent + normal) if u.tech == "wifi"]
        nru_pw  = [(u, max(1e-9, fair_weight(u))) for u, _ in (urgent + normal) if u.tech == "nru"]

        def distribute(pw, key):
            tot = sum(w for _, w in pw)
            if tot <= 0 or remain[key] <= 0: return
            for u, w in pw:
                tau[u.id] += remain[key] * (w / tot)

        distribute(wifi_pw, "wifi")
        distribute(nru_pw,  "nru")

        # Actual goodput (using final duty to estimate LBT loss)
        duty_wifi_used = sum(tau[u.id] for u in wifi_users)
        duty_nru_used  = sum(tau[u.id] for u in nru_users)
        loss_wifi = lbt_loss_fraction(duty_wifi_used, ch.wifi_busy, ch.wifi_lbt_base_fail) if wifi_users else 0.0
        loss_nru  = lbt_loss_fraction(duty_nru_used,  ch.nru_busy,  ch.nru_lbt_base_fail)  if nru_users  else 0.0

        for u in wifi_users + nru_users:
            eff = se_eff(u, ch)
            rate_raw = eff * ch.bw_hz * tau[u.id]
            loss = loss_wifi if u.tech == "wifi" else loss_nru
            goodput = rate_raw * (1.0 - loss)
            served = min(u.data_bits, goodput * env.epoch_s)
            per_user_served[u.id] += served

            power_w, _ = POWER_PROFILES[u.pwr_mode]
            e_per_bit = power_w / max(1e-6, eff)
            energy = e_per_bit * served

            rows.append({
                "alpha": alpha_fair, "channel": ch.id, "user": u.id, "tech": u.tech,
                "priority": u.priority, "pwr": u.pwr_mode, "battery": round(u.battery, 3),
                "latency_ms": u.latency_ms, "tau": round(tau[u.id], 6),
                "loss_frac": round(loss, 4),
                "rate_raw_bps": int(rate_raw),
                "goodput_bps": int(goodput),
                "served_bits": int(served),
                "energy_j": round(energy, 6),
            })

    df = pd.DataFrame(rows)
    df_user = df.groupby(["alpha", "user", "tech"], as_index=False)["served_bits"].sum()
    rates = (df_user["served_bits"] / env.epoch_s).to_numpy()
    rates = np.maximum(rates, 1e-6)
    if abs(alpha_fair - 1.0) < 1e-9:
        util = float(np.sum(np.log(rates)))
    else:
        util = float(np.sum((rates ** (1.0 - alpha_fair)) / (1.0 - alpha_fair)))
    return df, df_user, util

# -------------------------
# Demo generation
# -------------------------

def make_demo(n_wifi=16, n_nru=12, seed=2025) -> Tuple[List[User], Env]:
    random.seed(seed)
    np.random.seed(seed)
    users: List[User] = []
    for i in range(n_wifi):
        users.append(User(
            id=f"w{i+1}", tech="wifi",
            CQI=int(np.clip(np.random.normal(9, 3), 0, 15)),
            battery=float(np.clip(np.random.beta(2, 2), 0, 1)),
            data_bits=float(np.random.uniform(2e6, 12e6)),
            latency_ms=float(np.random.choice([15, 25, 40, 80, 120])),
            priority=random.choices(["emergency", "high", "normal", "bulk"], weights=[1, 3, 5, 3])[0],
            pwr_mode=random.choices(["low", "med", "high"], weights=[3, 5, 2])[0],
        ))
    for i in range(n_nru):
        users.append(User(
            id=f"n{i+1}", tech="nru",
            CQI=int(np.clip(np.random.normal(10, 3), 0, 15)),
            battery=float(np.clip(np.random.beta(2.5, 1.8), 0, 1)),
            data_bits=float(np.random.uniform(3e6, 15e6)),
            latency_ms=float(np.random.choice([10, 20, 35, 60, 120])),
            priority=random.choices(["emergency", "high", "normal", "bulk"], weights=[1, 3, 5, 3])[0],
            pwr_mode=random.choices(["low", "med", "high"], weights=[2, 5, 3])[0],
        ))
    channels = [
        Channel("ch37", bw_hz=160e6,
                wifi_busy=float(np.random.uniform(0.55, 0.85)),
                nru_busy=float(np.random.uniform(0.45, 0.80)),
                wifi_lbt_base_fail=float(np.random.uniform(0.06, 0.18)),
                nru_lbt_base_fail=float(np.random.uniform(0.04, 0.15))),
        Channel("ch39", bw_hz=160e6,
                wifi_busy=float(np.random.uniform(0.45, 0.80)),
                nru_busy=float(np.random.uniform(0.55, 0.85)),
                wifi_lbt_base_fail=float(np.random.uniform(0.05, 0.16)),
                nru_lbt_base_fail=float(np.random.uniform(0.06, 0.17))),
    ]
    env = Env(epoch_s=0.1, channels=channels)
    return users, env

# -------------------------
# Multi-epoch simulation functions
# -------------------------

def random_jitter(x: float, lo: float, hi: float, sigma: float) -> float:
    """Apply small jitter to busy or CQI, then clip to interval"""
    return float(np.clip(x + np.random.normal(0.0, sigma), lo, hi))

def step_env(users: List[User], env: Env,
             arrival_bits_mean: float,
             cqi_jitter: float = 0.4,
             busy_jitter: float = 0.03):
    """
    Evolve environment and traffic:
    - Each user gets new arrivals (normal distribution around arrival_bits_mean)
    - CQI jitters slightly
    - Channel busy and base LBT failure rates jitter slightly
    """
    # Arrivals & CQI jitter
    new_users = []
    for u in users:
        add_bits = max(0.0, np.random.normal(arrival_bits_mean, arrival_bits_mean*0.25))
        new_cqi = int(np.clip(np.random.normal(u.CQI, cqi_jitter), 0, 15))
        new_users.append(replace(u, data_bits=u.data_bits + add_bits, CQI=new_cqi))
    
    # Channel jitter
    new_chs = []
    for ch in env.channels:
        new_chs.append(Channel(
            id=ch.id,
            bw_hz=ch.bw_hz,
            wifi_busy=random_jitter(ch.wifi_busy, 0.0, 0.95, busy_jitter),
            nru_busy=random_jitter(ch.nru_busy, 0.0, 0.95, busy_jitter),
            wifi_lbt_base_fail=random_jitter(ch.wifi_lbt_base_fail, 0.01, 0.3, busy_jitter*0.5),
            nru_lbt_base_fail=random_jitter(ch.nru_lbt_base_fail, 0.01, 0.3, busy_jitter*0.5),
        ))
    return new_users, replace(env, channels=new_chs)

def one_epoch_allocate(users: List[User], env: Env, use_llm: bool, llm_model: str):
    """
    Run single epoch: choose policy, then do α-fairness allocation, return:
      - Updated users (with transmitted data subtracted)
      - Epoch statistics (throughput, energy, SLA hit rate)
    """
    state_json = build_state_json(users, env)
    if use_llm:
        try:
            llm_obj = llm_choose_policy(state_json, model=llm_model)
            policy = coerce_policy_from_llm(llm_obj, env)
        except Exception as e:
            print(f"[WARN] LLM failed, falling back to rule policy: {e}")
            policy = rule_policy_multi(env)
    else:
        policy = rule_policy_multi(env)

    # α selection (if LLM, use its choice; otherwise try 0/1/2 and pick best throughput)
    alphas = [policy.alpha_fair] if use_llm else [0.0, 1.0, 2.0]
    best = None
    for a in alphas:
        df, df_user, util = optimize_multi_alpha(users, env, policy, alpha_fair=a)
        energy = float(df["energy_j"].sum()) if "energy_j" in df.columns else 0.0
        bits = float(df["served_bits"].sum())
        
        # SLA hit rate calculation
        gp = df.groupby("user", as_index=False)["goodput_bps"].sum()
        hits = 0
        for u in users:
            need = latency_required_rate(u.data_bits, max(1.0, u.latency_ms))
            got = float(gp[gp["user"]==u.id]["goodput_bps"].sum()) if u.id in set(gp["user"]) else 0.0
            hits += (got >= need)
        sla = hits / max(1, len(users))
        
        cur = {"alpha": a, "df": df, "served_bits": bits, "energy": energy, "sla_hit_rate": sla}
        if (best is None) or (not use_llm and bits > best["served_bits"]):
            best = cur

    # Subtract transmitted data from user queues
    sent = best["df"].groupby("user", as_index=False)["served_bits"].sum()
    sent_map = {r["user"]: float(r["served_bits"]) for _, r in sent.iterrows()}
    new_users = [replace(u, data_bits=max(0.0, u.data_bits - sent_map.get(u.id, 0.0))) for u in users]

    return new_users, {
        "alpha_used": best["alpha"],
        "served_bits": best["served_bits"],
        "energy": best["energy"],
        "sla_hit_rate": best["sla_hit_rate"],
    }

def run_multi_epoch(epochs: int,
                    users: List[User], env: Env,
                    use_llm: bool, llm_model: str,
                    arrival_bits_mean: float,
                    cqi_jitter: float, busy_jitter: float):
    """
    Run multiple epochs continuously; return cumulative metrics and per-epoch curves.
    """
    users = [replace(u) for u in users]  # Deep copy
    env = replace(env, channels=[replace(ch) for ch in env.channels])

    stats = []
    total_bits = total_energy = 0.0
    
    for _ in range(epochs):
        users, env = step_env(users, env, arrival_bits_mean, cqi_jitter, busy_jitter)
        users, s = one_epoch_allocate(users, env, use_llm, llm_model)
        stats.append(s)
        total_bits += s["served_bits"]
        total_energy += s["energy"]
    
    bpj = (total_bits/total_energy) if total_energy > 0 else 0.0
    return {
        "per_epoch": stats,
        "total_bits": total_bits,
        "total_energy": total_energy,
        "bits_per_joule": bpj,
        "avg_sla_hit": float(np.mean([s["sla_hit_rate"] for s in stats])) if stats else 0.0,
    }

# -------------------------
# Main entry point
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Spectrum Agent with Alpha-Fairness + LLM Policy")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for policy selection")
    parser.add_argument("--llm-model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting")
    parser.add_argument("--export-prefix", type=str, default=".", help="Export directory")
    # Multi-epoch comparison
    parser.add_argument("--compare-multi", action="store_true",
                        help="Compare: No-LLM vs LLM in multi-epoch evolution")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--epoch-sec", type=float, default=0.1, help="Epoch duration (seconds)")
    parser.add_argument("--arrival-mbps", type=float, default=40.0, help="Mean arrival rate (Mbps)")
    parser.add_argument("--cqi-jitter", type=float, default=0.4, help="CQI jitter std dev")
    parser.add_argument("--busy-jitter", type=float, default=0.03, help="Channel busy jitter std dev")
    args = parser.parse_args()

    users, env = make_demo(seed=args.seed)
    env = replace(env, epoch_s=args.epoch_sec)
    outdir = args.export_prefix.rstrip("/\\")
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    if args.compare_multi:
        # Multi-epoch comparison path
        arrival_bits_mean = args.arrival_mbps * 1e6 * args.epoch_sec

        print("[RUN] No-LLM baseline (multi-epoch)...")
        res_rule = run_multi_epoch(args.epochs, users, env, use_llm=False, llm_model=args.llm_model,
                                  arrival_bits_mean=arrival_bits_mean,
                                  cqi_jitter=args.cqi_jitter, busy_jitter=args.busy_jitter)
        
        print("[RUN] LLM-assisted (multi-epoch)...")
        res_llm = run_multi_epoch(args.epochs, users, env, use_llm=True, llm_model=args.llm_model,
                                 arrival_bits_mean=arrival_bits_mean,
                                 cqi_jitter=args.cqi_jitter, busy_jitter=args.busy_jitter)

        # Save per-epoch CSVs
        df_rule = pd.DataFrame(res_rule["per_epoch"])
        df_llm = pd.DataFrame(res_llm["per_epoch"])
        path_rule = f"{outdir}/rule_per_epoch.csv"
        path_llm = f"{outdir}/llm_per_epoch.csv"
        df_rule.to_csv(path_rule, index=False)
        df_llm.to_csv(path_llm, index=False)
        print("\nCSV saved:")
        print(" ", os.path.abspath(path_rule))
        print(" ", os.path.abspath(path_llm))

        # Console comparison summary
        def fmt(x):
            try:
                return f"{x:,.2f}"
            except:
                return str(x)
        
        print("\n=== MULTI-EPOCH COMPARISON ===")
        print(f"Epochs={args.epochs} | epoch_s={args.epoch_sec} | arrival={args.arrival_mbps} Mb/s")
        print(f"{'Metric':<28s} {'No-LLM':>16s} {'LLM-Assisted':>16s}")
        print("-" * 60)
        print(f"{'Total Throughput (bits)':<28s} {fmt(res_rule['total_bits']):>16s} {fmt(res_llm['total_bits']):>16s}")
        print(f"{'Total Energy (J)':<28s} {fmt(res_rule['total_energy']):>16s} {fmt(res_llm['total_energy']):>16s}")
        print(f"{'Energy Efficiency (bits/J)':<28s} {fmt(res_rule['bits_per_joule']):>16s} {fmt(res_llm['bits_per_joule']):>16s}")
        print(f"{'Avg SLA Hit Rate (%)':<28s} {fmt(100*res_rule['avg_sla_hit']):>16s} {fmt(100*res_llm['avg_sla_hit']):>16s}")
        return

    # Single-epoch snapshot path (original logic)
    state_json = build_state_json(users, env)
    if args.use_llm:
        try:
            llm_obj = llm_choose_policy(state_json, model=args.llm_model)
            policy = coerce_policy_from_llm(llm_obj, env)
            policy.template = "llm_orchestrated"
        except Exception as e:
            print(f"[WARN] LLM policy failed, falling back to rule policy: {e}")
            policy = rule_policy_multi(env)
            policy.template = "rule_fallback"
    else:
        policy = rule_policy_multi(env)
        policy.template = "rule"

    alpha_list = [policy.alpha_fair] if args.use_llm else [0.0, 1.0, 2.0]
    results, per_user_tables, utils = [], [], []
    for a in alpha_list:
        df, df_user, util = optimize_multi_alpha(users, env, policy, alpha_fair=a)
        results.append(df.assign(alpha=a))
        df_user = df_user.copy()
        df_user["goodput_bps"] = df_user["served_bits"] / env.epoch_s
        per_user_tables.append(df_user.assign(alpha=a))
        utils.append((a, util))

    df_all = pd.concat(results, ignore_index=True)
    df_user_all = pd.concat(per_user_tables, ignore_index=True)
    alloc_path = f"{outdir}/allocations_by_alpha.csv"
    user_path = f"{outdir}/user_goodputs_by_alpha.csv"
    df_all.to_csv(alloc_path, index=False)
    df_user_all.to_csv(user_path, index=False)

    print("\nCSV saved:")
    print(" ", os.path.abspath(alloc_path))
    print(" ", os.path.abspath(user_path))

    if not args.no_plots:
        # Plot alpha-fairness utility
        xs = [a for a, _ in utils]
        ys = [u for _, u in utils]
        plt.figure()
        plt.plot(xs, ys, marker='o')
        plt.title("Alpha-fair Utility vs Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("Utility")
        plt.grid(True)
        plt.show()
        
        # Plot per-user goodput for each alpha
        for a in alpha_list:
            plt.figure()
            tmp = df_user_all[df_user_all["alpha"] == a].sort_values("goodput_bps", ascending=False)
            plt.bar(tmp["user"], tmp["goodput_bps"] / 1e6)
            plt.title(f"Per-user Goodput (alpha={a}) [Mb/s]")
            plt.xlabel("User")
            plt.ylabel("Goodput (Mb/s)")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()