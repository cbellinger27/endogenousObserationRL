import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import argparse

from dual_sensor_gridworld import DualSensorGridWorld

from amlr_q_agent_fine import AMLRQAgentFine
from amlr_q_agent_coarse import AMLRQAgentCoarse
from always_measure_agent import AlwaysMeasureAgent
from two_q_agent import TwoQAgentFine, TwoQAgentCoarse


# =========================================
# CONFIG
# =========================================
MAX_STEPS = 40
FRAME_STRIDE = 3   # ✅ every third frame

# =========================================
# CLI
# =========================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", type=str, default="baseline",
                        choices=["baseline", "amrl", "cascade_q"])

    parser.add_argument("--mode", type=str, default="fine",
                        choices=["coarse", "fine"])

    return parser.parse_args()


# =========================================
# LOAD AGENT
# =========================================
def load_agent(agent_class, path, agent_type):

    agent = agent_class()

    with open(path, "rb") as f:
        model = pickle.load(f)

    if agent_type in ["baseline", "amrl"]:
        agent.Q = model

    elif agent_type == "cascade":
        agent.Q_action = model["Q_action"]
        agent.Q_measure = model["Q_measure"]

    # disable exploration
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0

    return agent


# =========================================
# RENDER → IMAGE ARRAY
# =========================================
def render_to_image(env, sensor):

    env.render(sensor)

    fig = env.fig

    # ✅ force redraw and update
    fig.canvas.draw()
    fig.canvas.flush_events()

    img = np.array(fig.canvas.buffer_rgba())[:, :, :3].copy()

    return img



# =========================================
# RUN SINGLE EPISODE + CAPTURE FRAMES
# =========================================
def collect_frames(agent, env, agent_type):

    obs = env.reset()
    done = False
    step = 0

    frames = []
    sensors = []
    steps = []


    print("\n--- Running trajectory ---")

    while not done and step < MAX_STEPS:

        # ===============================
        # ACTION SELECTION
        # ===============================
        if agent_type == "baseline":
            action, sensor = agent.act(obs)

        elif agent_type == "amrl":
            action, sensor, _ = agent.act(obs)

        elif agent_type == "cascade":
            action, sensor, _ = agent.act(obs)

        # ===============================
        # STEP
        # ===============================
        next_obs, reward, done, info = env.step(action, sensor)

        print(
            f"Step {step:02d} | Pos={env.agent_pos} | "
            f"A={action} | S={sensor}"
        )

        # ===============================
        # STORE FRAME
        # ===============================
        img = render_to_image(env, sensor)
        frames.append(img)
        sensors.append(sensor)
        steps.append(step)

        obs = next_obs
        step += 1

    return frames, sensors, steps


# =========================================
# MAKE 1xN PANEL (EVERY THIRD FRAME)
# =========================================
def make_panel(frames, sensors, steps, output_file="trajectory_panel.png"):

    # # ✅ subsample consistently
    frames = frames[::FRAME_STRIDE]
    sensors = sensors[::FRAME_STRIDE]
    steps = steps[::FRAME_STRIDE]
    
    # frames = frames[1:-1]
    # sensors = sensors[1:-1]
    # steps = steps[1:-1]
     
    N = len(frames)

    fig, axes = plt.subplots(1, N, figsize=(3*N, 3))

    if N == 1:
        axes = [axes]

    for i in range(N):

        axes[i].imshow(frames[i])
        axes[i].axis("off")

        axes[i].set_title(
            f"t={steps[i]}",
            fontsize=12
        )
        plt.subplots_adjust(        
            left=0.01,        
            right=0.99,        
            top=0.85,        
            bottom=0.05,        
            wspace=0.02,   # ✅ VERY small horizontal gap        hspace=0.0    
            )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# =========================================
# MAIN
# =========================================
def main():

    random.seed(1)
    np.random.seed(1)

    args = parse_args()

    if args.agent in ["baseline"]:
        agent_class = AlwaysMeasureAgent
        path = "saved_models/Baseline_Always_best.pkl" 
        agent_type = "baseline"
        output_file= "Baseline_Always_best"

    if args.agent in ["amrl"]:
        if args.mode == "coarse":
            agent_class = AMLRQAgentCoarse
            path = "saved_models/AMLR_Coarse_best.pkl" 
            agent_type = "amrl"
            output_file= "AMLR_Coarse_best"
        else:
            agent_class = AMLRQAgentFine
            path = "saved_models/AMLR_Fine_best.pkl"
            agent_type = "amrl"
            output_file= "AMLR_Fine_best"

    if args.agent in ["cascade_q", "all"]:
        if args.mode == "coarse":
            agent_class = TwoQAgentCoarse
            path = "saved_models/CASCADE_Q_Coarse_best.pkl" 
            agent_type = "cascade"
            output_file= "CASCADE_Q_Coarse_best"
        else:
            agent_class = TwoQAgentFine
            path = "saved_models/CASCADE_Q_Fine_best.pkl" 
            agent_type = "cascade"
            output_file= "CASCADE_Q_Fine_best"


    print(f"\nRunning {agent_type}")


    agent = load_agent(agent_class, path, agent_type)
    env = DualSensorGridWorld(include_time=False)

    frames, sensors, steps = collect_frames(agent, env, agent_type)

    make_panel(
        frames,
        sensors,
        steps,
        output_file=f"{output_file}_panel.png"
    )


if __name__ == "__main__":
    main()
