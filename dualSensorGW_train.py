import numpy as np
import random
import os
import pickle
import argparse
import matplotlib.pyplot as plt

from dual_sensor_gridworld import DualSensorGridWorld

from amlr_q_agent_coarse import AMLRQAgentCoarse
from amlr_q_agent_fine import AMLRQAgentFine
from always_measure_agent import AlwaysMeasureAgent
from two_q_agent import TwoQAgentFine, TwoQAgentCoarse


# =========================================
# CONFIG
# =========================================
NUM_SEEDS = 5
EPISODES = 50000
MAX_STEPS = 200


# =========================================
# CLI
# =========================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", type=str, default="all",
                        choices=["baseline", "amlr", "cascade_q", "all"])

    parser.add_argument("--mode", type=str, default="fine",
                        choices=["coarse", "fine"])

    return parser.parse_args()


# =========================================
# TRAIN
# =========================================
def train(env, agent, agent_type):

    logs = {
        "reward": [],
        "return": [],
        "cost": [],
        "measure_total": [],
        "measure_local": [],
        "measure_global": [],
        "success": [],
        "hazard": [],
        "state_counts": np.zeros((env.size, env.size)),
        "local_counts": np.zeros((env.size, env.size)),
        "global_counts": np.zeros((env.size, env.size))
    }

    for ep in range(EPISODES):

        obs = env.reset()
        done = False
        step = 0

        total_reward = 0
        total_cost = 0
        G = 0

        m_total = 0
        m_local = 0
        m_global = 0

        reached_goal = False
        hit_hazard = False

        while not done and step < MAX_STEPS:

            # ========= ACTION =========
            if agent_type == "baseline":
                action, sensor = agent.act(obs)

            elif agent_type == "amlr":
                action, sensor, s_idx = agent.act(obs)

            elif agent_type == "twoq":
                action, sensor, m_idx = agent.act(obs)

            # ========= STEP =========
            next_obs, reward, done, info = env.step(action, sensor)

            total_reward += info["ext_reward"]
            total_cost += info["cost"]
            G += reward

            # ========= EVENT TRACKING =========
            if "event" in info:
                if info["event"] == "goal":
                    reached_goal = True
                elif info["event"] == "hazard":
                    hit_hazard = True

            # ========= STATE TRACK =========
            ax, ay = env.agent_pos
            logs["state_counts"][ay, ax] += 1

            # ========= SENSOR TRACK =========
            if sensor == 1:
                m_local += 1
                m_total += 1
                logs["local_counts"][ay, ax] += 1
            elif sensor == 2:
                m_global += 1
                m_total += 1
                logs["global_counts"][ay, ax] += 1
            elif sensor == 3:
                m_local += 1
                m_global += 1
                m_total += 2
                logs["local_counts"][ay, ax] += 1
                logs["global_counts"][ay, ax] += 1

            # ========= UPDATE =========
            if agent_type == "baseline":
                agent.update(obs, action, reward, next_obs, done)

            elif agent_type == "amlr":
                agent.update(obs, s_idx, action, reward, next_obs, done)

            elif agent_type == "twoq":
                agent.update(
                    obs,
                    action,
                    m_idx,
                    info["ext_reward"],
                    info["cost"],
                    next_obs,
                    done
                )

            obs = next_obs
            step += 1

        # ========= EPISODE LOG =========
        logs["reward"].append(total_reward)
        logs["return"].append(G)
        logs["cost"].append(total_cost)
        logs["measure_total"].append(m_total)
        logs["measure_local"].append(m_local)
        logs["measure_global"].append(m_global)
        logs["success"].append(1 if reached_goal else 0)
        logs["hazard"].append(1 if hit_hazard else 0)

        if (ep + 1) % 5000 == 0:
            print(f"{agent_type} | Ep {ep+1} | Return {G:.2f} | Cost {total_cost:.2f}")

    return logs


# =========================================
# RUN MULTI-SEED
# =========================================
def run_experiment(name, agent_class, agent_type):

    print(f"\n==== Running {name} ====")

    all_logs = []
    best_score = -1e9
    best_model = None

    for seed in range(NUM_SEEDS):

        print(f"{name} | Seed {seed}")

        np.random.seed(seed)
        random.seed(seed)

        env = DualSensorGridWorld(include_time=False)

        if agent_type == "twoq":
            agent = agent_class(cost_weight=1)
        else:
            agent = agent_class()

        logs = train(env, agent, agent_type)
        all_logs.append(logs)

        score = np.mean(logs["return"][-100:])

        print(f"{name} | Seed {seed} | Last100 Return {score:.2f}")

        if score > best_score:
            best_score = score

            if agent_type in ["baseline", "amlr"]:
                best_model = agent.Q.copy()
            elif agent_type == "twoq":
                best_model = {
                    "Q_action": agent.Q_action.copy(),
                    "Q_measure": agent.Q_measure.copy()
                }

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("saved_data", exist_ok=True)

    # Save best model
    save_obj = {
        "model": best_model,
        "score": best_score,
        "config": {
            "episodes": EPISODES,
            "seeds": NUM_SEEDS,
            "agent": name
        }
    }

    with open(f"saved_models/{name}_best.pkl", "wb") as f:
        pickle.dump(save_obj, f)

    # ✅ SAVE FULL LOGS
    with open(f"saved_data/{name}_logs.pkl", "wb") as f:
        pickle.dump(all_logs, f)

    print(f"Saved logs → saved_data/{name}_logs.pkl")

    return all_logs


# =========================================
# AGGREGATION
# =========================================
def aggregate(all_logs, key):
    filtered = [log[key] for log in all_logs if key in log]

    if len(filtered) == 0:
        raise ValueError(f"No logs contain key: {key}")

    data = np.array(filtered)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ci = 1.96 * std / np.sqrt(len(filtered))
    return mean, ci


def aggregate_rate(all_logs, key):
    return aggregate(all_logs, key)


# =========================================
# PLOTTING
# =========================================
def plot_all(results):

    import os
    os.makedirs("saved_plots", exist_ok=True)

    plt.style.use("seaborn-v0_8-paper")

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 10,
        "lines.linewidth": 2.5
    })

    colors = {
        "Baseline_Always": "#1f77b4",
        "AMLR_Coarse": "#ff7f0e",
        "AMLR_Fine": "#2ca02c",
        "CASCADE_Q_Coarse": "#d62728",
        "CASCADE_Q_Fine": "#9467bd"
    }

    x = np.arange(EPISODES)

    def smooth(y, w=200):
        return np.convolve(y, np.ones(w)/w, mode="same")

    # ======================================================
    # ✅ 1. RETURN + COST (SAME PLOT)
    # ======================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, logs in results.items():

        r_mean, r_ci = aggregate(logs, "return")
        c_mean, c_ci = aggregate(logs, "cost")

        r_mean, r_ci = smooth(r_mean), smooth(r_ci)
        c_mean, c_ci = smooth(c_mean), smooth(c_ci)

        ax.plot(x, r_mean, color=colors[name], label=f"{name} (Return)")
        ax.fill_between(x, r_mean-r_ci, r_mean+r_ci,
                        color=colors[name], alpha=0.15)

        ax.plot(x, c_mean, linestyle="-.", label=f"{name} (Cost)")
        ax.fill_between(x, c_mean-c_ci, c_mean+c_ci,alpha=0.15)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.set_title("Return and Cost")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig(f"saved_plots/fig_return_cost_{name}.png", dpi=300)

    # ======================================================
    # ✅ 2. SENSING (ALL IN ONE)
    # ======================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    sensing_types = [
        ("measure_total", "-", "Total"),
        ("measure_local", "--", "Local"),
        ("measure_global", ":", "Global")
    ]

    for name, logs in results.items():

        for key, style, label_name in sensing_types:
            mean, ci = aggregate(logs, key)
            mean, ci = smooth(mean), smooth(ci)

            ax.plot(x, mean, linestyle=style, color=colors[name],
                    label=f"{name} ({label_name})")
            ax.fill_between(x, mean-ci, mean+ci,
                        color=colors[name], alpha=0.15)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.set_title("Sensing Behavior")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig(f"saved_plots/fig_sensing_{name}.png", dpi=300)

    # ======================================================
    # ✅ 3. SUCCESS + HAZARD (SAME PLOT ✅ CORRECT)
    # ======================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, logs in results.items():

        s_mean, s_ci = aggregate(logs, "success")
        h_mean, h_ci = aggregate(logs, "hazard")

        s_mean, s_ci = smooth(s_mean), smooth(s_ci)
        h_mean, h_ci = smooth(h_mean), smooth(h_ci)

        ax.plot(x, s_mean, color=colors[name],
                label=f"{name} (Success)")
        ax.fill_between(x, s_mean-s_ci, s_mean+s_ci,
                        color=colors[name], alpha=0.15)

        ax.plot(x, h_mean, linestyle="-.",label=f"{name} (Hazard)")
        ax.fill_between(x, h_mean-h_ci, h_mean+h_ci, alpha=0.15)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_title("Success vs Hazard")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig(f"saved_plots/fig_success_hazard_{name}.png", dpi=300)

# =========================================
# MAIN
# =========================================
def main():

    args = parse_args()
    results = {}

    if args.agent in ["baseline", "all"]:
        results["Baseline_Always"] = run_experiment(
            "Baseline_Always",
            AlwaysMeasureAgent,
            "baseline"
        )

    if args.agent in ["amlr", "all"]:
        if args.mode == "coarse":
            results["AMLR_Coarse"] = run_experiment(
                "AMLR_Coarse",
                AMLRQAgentCoarse,
                "amlr"
            )
        else:
            results["AMLR_Fine"] = run_experiment(
                "AMLR_Fine",
                AMLRQAgentFine,
                "amlr"
            )

    if args.agent in ["cascade_q", "all"]:
        if args.mode == "coarse":
            results["CASCADE_Q_Coarse"] = run_experiment(
                "CASCADE_Q_Coarse",
                TwoQAgentCoarse,
                "twoq"
            )
        else:
            results["CASCADE_Q_Fine"] = run_experiment(
                "CASCADE_Q_Fine",
                TwoQAgentFine,
                "twoq"
            )

    if results:
        plot_all(results)


if __name__ == "__main__":
    main()
