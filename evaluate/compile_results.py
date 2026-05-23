import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

NUM_SEEDS = 5


# =========================================
# CLI: SELECT METHODS
# =========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="List of methods to include (e.g. CASCADE_Q_Fine Baseline_Always)"
    )
    return parser.parse_args()


# =========================================
# LOAD
# =========================================
def load_all_logs(folder="saved_data"):

    results = {}

    for file in os.listdir(folder):
        print(file)
        if file.endswith(".pkl"):
            name = file.replace("_logs.pkl", "")
            with open(os.path.join(folder, file), "rb") as f:
                results[name] = pickle.load(f)

    return results


# =========================================
# FILTER METHODS
# =========================================
def filter_methods(results, selected):

    if selected is None:
        return results

    filtered = {}
    for name in selected:
        if name in results:
            filtered[name] = results[name]
        else:
            print(f"Warning: {name} not found in saved data")

    return filtered


# =========================================
# COLOR HANDLING (AUTO DISTINCT)
# =========================================
def assign_colors(methods):

    cmap = plt.get_cmap("tab10")

    colors = {}
    for i, name in enumerate(methods):
        colors[name] = cmap(i % 10)

    return colors


# =========================================
# AGGREGATE
# =========================================
def aggregate(all_logs, key):

    filtered = [log[key] for log in all_logs if key in log]

    if len(filtered) == 0:
        return None, None

    data = np.array(filtered)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ci = 1.96 * std / np.sqrt(len(filtered))

    return mean, ci


# =========================================
# SMOOTH
# =========================================
def smooth(y, w=200):
    if y is None:
        return None
    return np.convolve(y, np.ones(w)/w, mode="same")


# =========================================
# PLOTTING
# =========================================
def plot_all(results):

    os.makedirs("saved_plots", exist_ok=True)

    plt.style.use("seaborn-v0_8-paper")

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 10,
        "lines.linewidth": 2.5
    })

    methods = list(results.keys())
    colors = assign_colors(methods)

    # infer episode length
    ep_len = len(results[methods[0]][0]["return"])
    x = np.arange(ep_len)

    methods_str = "".join(methods)


    # =========================================
    # 1. RETURN + COST
    # =========================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in methods:
        logs = results[name]

        r, r_ci = aggregate(logs, "return")
        c, c_ci = aggregate(logs, "cost")

        if r is None or c is None:
            continue
        
        r = smooth(r)
        r_ci = smooth(r_ci)
        c = smooth(c)
        c_ci = smooth(c_ci)

        ax.plot(x, r, color=colors[name], label=f"{name} (Return)")
        ax.fill_between(x, r-r_ci, r+r_ci,color=colors[name], alpha=0.15)

        ax.plot(x, c, linestyle="--", color=colors[name],label=f"{name} (Cost)")
        ax.fill_between(x, c-c_ci, c+c_ci,color=colors[name], alpha=0.15)

    ax.set_title("Return and Cost")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.set_xlim(0,49900)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig(f"saved_plots/paper_return_cost{methods_str}.png", dpi=300)

    # =========================================
    # 2. SENSING
    # =========================================
    fig, ax = plt.subplots(figsize=(8, 5))

    sensing_keys = [
        ("measure_total", "-", "Total"),
        ("measure_local", "--", "Local"),
        ("measure_global", ":", "Global")
    ]

    
    for name in methods:
        logs = results[name]

        for key, style, label_name in sensing_keys:

            m, m_ci = aggregate(logs, key)
            if m is None:
                continue

            m = smooth(m)
            m_ci = smooth(m_ci)

            ax.plot(x, m, linestyle=style, color=colors[name],
                    label=f"{name} ({label_name})")
            ax.fill_between(x, m-m_ci, m+m_ci,color=colors[name], alpha=0.15)

    ax.set_title("Sensing Behavior")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.set_xlim(0,49900)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig(f"saved_plots/paper_sensing{methods_str}.png", dpi=300)

    # =========================================
    # 3. SUCCESS + HAZARD
    # =========================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in methods:
        logs = results[name]

        s, s_ci = aggregate(logs, "success")
        h, h_ci = aggregate(logs, "hazard")

        if s is None or h is None:
            continue

        s = smooth(s)
        h = smooth(h)
        s_ci = smooth(s_ci)
        h_ci = smooth(h_ci)

        ax.plot(x, s, color=colors[name], label=f"{name} (Success)")
        ax.fill_between(x, s-s_ci, s+s_ci,color=colors[name], alpha=0.15)
        ax.plot(x, h, linestyle="--",
                color=colors[name], label=f"{name} (Hazard)")
        ax.fill_between(x, h-h_ci, h+h_ci,color=colors[name], alpha=0.15)

    ax.set_ylim(0, 1)
    ax.set_title("Success vs Hazard Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_xlim(0,49900)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig(f"saved_plots/paper_success_hazard{methods_str}.png", dpi=300)

    # # =========================================
    # # 4. STEPS
    # # =========================================
    # fig, ax = plt.subplots(figsize=(8, 5))

    # for name in methods:
    #     logs = results[name]

    #     st, _ = aggregate(logs, "steps")
    #     if st is None:
    #         continue

    #     st = smooth(st)

    #     ax.plot(x, st, color=colors[name], label=name)

    # ax.set_title("Episode Length")
    # ax.set_xlabel("Episode")
    # ax.set_ylabel("Steps")
    # ax.grid(alpha=0.3)
    # ax.legend()

    # plt.tight_layout()
    # plt.savefig("saved_plots/paper_steps.png", dpi=300)

    plt.show()


# =========================================
# MAIN
# =========================================
def main():

    args = parse_args()

    results = load_all_logs("saved_data")

    if len(results) == 0:
        print("No data found.")
        return

    results = filter_methods(results, args.methods)

    if len(results) == 0:
        print("No valid methods selected.")
        return

    plot_all(results)


if __name__ == "__main__":
    main()
