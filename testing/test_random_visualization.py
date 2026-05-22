import numpy as np
import random
import cv2
import os

from dual_sensor_gridworld import DualSensorGridWorld
from random_agent import RandomAgent


# =========================================
# CONFIG
# =========================================
EPISODES = 5
MAX_STEPS = 20
SAVE_VIDEO = True
FPS = 5


# =========================================
# RENDER → IMAGE
# =========================================
def render_to_image(env, sensor):

    env.render(sensor)

    fig = env.fig
    fig.canvas.draw()

    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

    return img


# =========================================
# SAVE VIDEO
# =========================================
def save_video(frames, filename):

    if len(frames) == 0:
        return

    h, w, _ = frames[0].shape

    writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h)
    )

    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved video: {filename}")


# _________________________________________
# MAIN LOOP
# ________________________________________
def main():

    random.seed(0)
    np.random.seed(0)

    env = DualSensorGridWorld(include_time=False)
    agent = RandomAgent()

    frames = []

    for ep in range(EPISODES):

        print(f"\nEpisode {ep+1}")

        obs = env.reset()
        done = False
        step = 0

        while not done and step < MAX_STEPS:

            action, sensor = agent.act(obs)

            next_obs, reward, done, info = env.step(action, sensor)

            print(
                f"Step {step:2d} | "
                f"A={action} | S={sensor} | "
                f"R={info['ext_reward']:.2f} | "
                f"C={info['cost']:.3f} | "
                f"Collision={info['collision']}"
            )

            frame = render_to_image(env, sensor)
            frames.append(frame)

            obs = next_obs
            step += 1

    # =========================================
    # SAVE VIDEO
    # =========================================
    if SAVE_VIDEO:
        os.makedirs("videos", exist_ok=True)
        save_video(frames, "videos/random_agent.mp4")


if __name__ == "__main__":
    main()
