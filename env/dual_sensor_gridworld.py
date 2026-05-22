import numpy as np
import random
import matplotlib.pyplot as plt

class DualSensorGridWorld:

    def __init__(self,
                 size=6,
                 stochasticity=0.0,
                 terminate_on_hazard=True,
                 include_time=False,
                 global_cost=1,
                 local_cost=0.002,
                 local_radius=3,
                 local_noise=0.1):

        self.size = size
        self.stochasticity = stochasticity
        self.terminate_on_hazard = terminate_on_hazard
        self.include_time = include_time

        self.global_cost = global_cost
        self.local_cost = local_cost
        self.local_radius = local_radius
        self.local_noise = local_noise
        self.timestep = 0

        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

    # =========================================
    # RESET
    # =========================================
    def reset(self):

        self.agent_pos = self.start
        self._place_hazard()

        # ✅ NEW: memory state
        self.last_hazard = (0, 0)
        self.prev_sensor = 0
        self.timestep = 0

        return self._get_obs(0)

    # =========================================
    # HAZARD
    # =========================================
    def _place_hazard(self):

        while True:
            x = random.randint(1, self.size - 2)
            y = random.randint(1, self.size - 2)

            cells = [(x + dx, y + dy)
                     for dx in [-1, 0, 1]
                     for dy in [-1, 0, 1]]

            if self.start not in cells and self.goal not in cells:
                self.hazard_cells = cells
                self.hazard_center = (x, y)
                break

    # =========================================
    # STEP
    # =========================================
    def step(self, action, sensor_type):

        self.timestep += 1
        epTotalSteps = 0
        event = ""
        if random.random() < self.stochasticity:
            action = random.randint(0, 3)

        x, y = self.agent_pos

        moves = {
            0: (x, y + 1),
            1: (x, y - 1),
            2: (x - 1, y),
            3: (x + 1, y)
        }

        nx, ny = moves[action]
        nx = max(0, min(self.size - 1, nx))
        ny = max(0, min(self.size - 1, ny))

        reward = -0.01
        done = False
        collision = False
        atGoal = False

        if (nx, ny) in self.hazard_cells:
            reward -= 15
            collision = True
            event = "hazard"
            if self.terminate_on_hazard:
                done = True
                epTotalSteps = self.timestep

        self.agent_pos = (nx, ny)

        if self.agent_pos == self.goal:
            done = True
            atGoal = True
            event = "goal"
            epTotalSteps = self.timestep

        ax, ay = self.agent_pos
        hx, hy = self.hazard_center

        dist = abs(hx - ax) + abs(hy - ay)

        # =====================================
        # SENSING
        # =====================================
        cost = 0

        if sensor_type == 2:   # global
            hx_rel = hx - ax
            hy_rel = hy - ay
            cost = self.global_cost

            # ✅ store measurement
            self.last_hazard = (hx, hy)

        elif sensor_type == 1:  # local
            cost = self.local_cost

            if dist <= self.local_radius:
                hx_rel = (hx - ax) + np.random.normal(0, self.local_noise)
                hy_rel = (hy - ay) + np.random.normal(0, self.local_noise)

                self.last_hazard = (hx, hy)
            else:
                hx_rel, hy_rel = 0, 0

        elif sensor_type == 3:  # both
            hx_rel = hx - ax
            hy_rel = hy - ay
            cost = self.global_cost + self.local_cost

            self.last_hazard = (hx, hy)

        else:  # none
            hx_rel, hy_rel = 0, 0

        # =====================================
        # USE LAST MEASUREMENT
        # =====================================
        lhx, lhy = self.last_hazard
        last_hx_rel = lhx - ax
        last_hy_rel = lhy - ay

        total_reward = reward - cost

        obs = np.array([
            ax,
            ay,
            hx_rel,
            hy_rel,
            self.goal[0] - ax,
            self.goal[1] - ay,
            self.prev_sensor,    # ✅ NEW
            last_hx_rel,         # ✅ NEW
            last_hy_rel          # ✅ NEW
        ])

        # update previous sensor AFTER building obs
        self.prev_sensor = sensor_type

        return obs, total_reward, done, {
            "ext_reward": reward,
            "cost": cost,
            "is_goal": atGoal,
            "is_hazard": collision,
            "total_steps": epTotalSteps,
            "event": event

        }

    # =========================================
    # OBS
    # =========================================
    def _get_obs(self, sensor_type):

        ax, ay = self.agent_pos
        gx, gy = self.goal

        lhx, lhy = getattr(self, "last_hazard", (0, 0))
        last_hx_rel = lhx - ax
        last_hy_rel = lhy - ay

        return np.array([
            ax,
            ay,
            0,
            0,
            gx - ax,
            gy - ay,
            self.prev_sensor,
            last_hx_rel,
            last_hy_rel
        ])
    
    # =========================================
    # RENDER
    # =========================================
    def render(self, sensor_type=0):

        if not hasattr(self, "fig"):
            self.fig, self.ax = plt.subplots(figsize=(5, 5))

        self.ax.clear()

        # -------------------------------------
        # GRID BACKGROUND
        # -------------------------------------
        grid = np.zeros((self.size, self.size))

        # hazard cells
        for (x, y) in self.hazard_cells:
            grid[y, x] = 0.4

        # goal
        gx, gy = self.goal
        grid[gy, gx] = 0.8

        # agent
        ax_pos, ay_pos = self.agent_pos
        grid[ay_pos, ax_pos] = 1.0

        self.ax.imshow(grid, origin="lower", cmap="viridis")

        # -------------------------------------
        # DRAW ELEMENTS
        # -------------------------------------

        # ✅ hazards
        for hx, hy in self.hazard_cells:
            self.ax.scatter(hx, hy, marker="s", color="black", s=120)

        # ✅ goal
        self.ax.scatter(gx, gy, marker="*", color="purple", s=200)

        # ✅ start
        self.ax.scatter(self.start[0], self.start[1],
                        marker="o", color="gold", s=120)

        # ✅ agent (color based on current sensing)
        col = _sensor_color(sensor_type)
        self.ax.scatter(ax_pos, ay_pos,
                        marker="o",
                        color=col,
                        s=150,
                        edgecolors="black")

        # -------------------------------------
        # ✅ LAST OBSERVED HAZARD (MEMORY)
        # -------------------------------------
        lhx, lhy = self.last_hazard
        self.ax.scatter(lhx, lhy,
                        marker="x",
                        color="white",
                        s=200,
                        linewidths=2,
                        label="last measurement")

        # -------------------------------------
        # GRID STYLE
        # -------------------------------------
        self.ax.set_xticks(range(self.size))
        self.ax.set_yticks(range(self.size))
        self.ax.grid(True)

        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(-0.5, self.size - 0.5)

        # -------------------------------------
        # TEXT INFO
        # -------------------------------------
        self.ax.set_title(
            f"Sensor: {sensor_type} | Prev: {self.prev_sensor}"
        )

        # -------------------------------------
        # LEGEND
        # -------------------------------------
        legend_elements = [
            plt.Line2D([0],[0], marker='o', color='w',
                    label='None', markerfacecolor='gray', markersize=10),

            plt.Line2D([0],[0], marker='o', color='w',
                    label='Local', markerfacecolor='blue', markersize=10),

            plt.Line2D([0],[0], marker='o', color='w',
                    label='Global', markerfacecolor='red', markersize=10),

            plt.Line2D([0],[0], marker='o', color='w',
                    label='Both', markerfacecolor='green', markersize=10),

            plt.Line2D([0],[0], marker='x', color='white',
                    label='Last hazard', markersize=10)
        ]

        self.ax.legend(handles=legend_elements,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=5,
                    fontsize=8)

        plt.pause(0.1)
        
# =========================================
# SENSOR COLOR
# =========================================
def _sensor_color(sensor):

    if sensor == 3:
        return "green"      # both
    elif sensor == 2:
        return "red"        # global
    elif sensor == 1:
        return "blue"       # local
    else:
        return "gray"       # none
