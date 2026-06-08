import os
import csv
import math
import numpy as np
import pygame

class Visualiser:
    """
    Handles all rendering and logging for both simulator and real robot.
    Supports:
      - Map/occupancy rendering
      - Ground truth (when available)
      - Odometry
      - State estimates + uncertainty
      - Unified CSV logging
    """
    def __init__(self, world_width, world_height, goal_pos, render_mode=None, window_size=(600,600), occupancy_grid=None, grid_resolution=0.1, csv_path=None):
        self.world_width = world_width
        self.world_height = world_height
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.render_mode = render_mode
        self.window_size = window_size
        self.occupancy_grid = occupancy_grid
        self.grid_resolution = grid_resolution
        self.csv_path = csv_path
        self._file = None
        self._writer = None
        self._columns = [
            "est_x", "est_y", "odom_x", "odom_y", "gt_x", "gt_y",
            "cov_x", "cov_y", "cov_xy", "cov_yx",
            "heading_cmd", "speed_cmd", "heading", "speed",
            "setpoint_x", "setpoint_y"
        ]
        self._gt_traj = []
        self._odom_traj = []
        self._est_traj = []
        self._overlay_real_traj = None
        self._overlay_odom_traj = None
        self._overlay_true_traj = None
        self.belief_mean = None
        self.belief_cov = None
        self.visual_occupancy_grid = None
        self.distance_map = None
        self.screen = None
        self.clock = None

    def start_logging(self):
        if not self.csv_path or self._writer is not None:
            return
        dirpath = os.path.dirname(self.csv_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._columns)
        self._file.flush()

    def stop_logging(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

    def reset(self):
        self._gt_traj.clear()
        self._odom_traj.clear()
        self._est_traj.clear()
        self._overlay_real_traj = None
        self._overlay_odom_traj = None
        self._overlay_true_traj = None
        self.belief_mean = None
        self.belief_cov = None

    def set_belief(self, mean, cov):
        self.belief_mean = np.asarray(mean, dtype=np.float32)
        self.belief_cov = np.asarray(cov, dtype=np.float32)

    def set_goal(self, goal_pos):
        self.goal_pos = np.array(goal_pos, dtype=np.float32)

    def set_occupancy(self, grid, resolution, visual_grid=None):
        self.occupancy_grid = grid
        self.grid_resolution = resolution
        if visual_grid is not None:
            self.visual_occupancy_grid = visual_grid

    def set_distance_map(self, dist_map):
        self.distance_map = dist_map

    def set_overlay_trajectories(self, real=None, odom=None, true=None):
        def _clean(traj):
            if traj is None:
                return None
            arr = np.asarray(traj, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("Trajectory must have shape (N,2)")
            return arr

        self._overlay_real_traj = _clean(real)
        self._overlay_odom_traj = _clean(odom)
        self._overlay_true_traj = _clean(true)

    def record(self, gt_state, odom_state, action, est_state=None, est_cov=None, setpoint=None):
        # Append to trajectories
        if gt_state is not None:
            self._gt_traj.append(tuple(gt_state[:2]))
        if odom_state is not None:
            self._odom_traj.append(tuple(odom_state[:2]))
        if est_state is not None:
            self._est_traj.append(tuple(est_state[:2]))
        # Prepare log row
        row = {
            "gt_x": float(gt_state[0]) if gt_state is not None else np.nan,
            "gt_y": float(gt_state[1]) if gt_state is not None else np.nan,
            "odom_x": float(odom_state[0]) if odom_state is not None else np.nan,
            "odom_y": float(odom_state[1]) if odom_state is not None else np.nan,
            "est_x": float(est_state[0]) if est_state is not None else np.nan,
            "est_y": float(est_state[1]) if est_state is not None else np.nan,
            "heading_cmd": float(action[1]) if action is not None else np.nan,
            "speed_cmd": float(action[0]) if action is not None else np.nan,
            "heading": float(odom_state[2]) if odom_state is not None and len(odom_state) > 2 else np.nan,
            "speed": float(odom_state[3]) if odom_state is not None and len(odom_state) > 3 else np.nan,
        }
        if setpoint is not None:
            setpoint_arr = np.asarray(setpoint, dtype=np.float32).reshape(-1)
            if setpoint_arr.shape[0] >= 2:
                row["setpoint_x"] = float(setpoint_arr[0])
                row["setpoint_y"] = float(setpoint_arr[1])
            else:
                row["setpoint_x"] = np.nan
                row["setpoint_y"] = np.nan
        else:
            row["setpoint_x"] = np.nan
            row["setpoint_y"] = np.nan
        # Covariance
        if est_cov is not None:
            cov = np.asarray(est_cov, dtype=np.float32)
            if cov.shape == (4, 4):
                cov2 = cov[0:2, 0:2]
            elif cov.shape == (2, 2):
                cov2 = cov
            else:
                cov2 = np.full((2,2), np.nan)
            row["cov_x"] = float(cov2[0,0])
            row["cov_y"] = float(cov2[1,1])
            row["cov_xy"] = float(cov2[0,1])
            row["cov_yx"] = float(cov2[1,0])
        else:
            row["cov_x"] = row["cov_y"] = row["cov_xy"] = row["cov_yx"] = np.nan
        # Write
        if self._writer is not None:
            self._writer.writerow([row.get(col, np.nan) for col in self._columns])
            self._file.flush()

    def get_traj(self):
        return [x for x, _ in self._odom_traj], [y for _, y in self._odom_traj], None

    def _init_pygame(self):
        if self.screen is not None:
            return
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Sphero Visualiser")
        self.clock = pygame.time.Clock()

    def render(self, gt_state, odom_state):
        if self.render_mode != "human":
            return
        self._init_pygame()
        width, height = self.window_size
        cx, cy = width // 2, height // 2
        # Colors
        bg_color = (20, 20, 30)
        axis_color = (80, 80, 80)
        border_color = (100, 100, 255)
        true_color = (0, 200, 0)
        odom_color = (50, 150, 255)
        goal_color = (255, 215, 0)
        obstacle_color = (200, 200, 200)
        belief_color = (255, 0, 255)
        heading_len = 0.15
        self.screen.fill(bg_color)
        # Axes
        pygame.draw.line(self.screen, axis_color, (0, cy), (width, cy), 1)
        pygame.draw.line(self.screen, axis_color, (cx, 0), (cx, height), 1)
        # Scaling
        scale_x = width / self.world_width
        scale_y = height / self.world_height
        scale = min(scale_x, scale_y) * 0.9
        rect_w = self.world_width * scale
        rect_h = self.world_height * scale
        rect = pygame.Rect(0, 0, rect_w, rect_h)
        rect.center = (cx, cy)
        pygame.draw.rect(self.screen, border_color, rect, 1)
        def world_to_screen(xw, yw):
            px = cx + xw * scale
            py = cy - yw * scale
            return int(px), int(py)
        def draw_polyline(traj, color, width=2):
            if traj is None or len(traj) < 2:
                return
            pts = [world_to_screen(float(x), float(y)) for x, y in traj]
            pygame.draw.lines(self.screen, color, False, pts, width)
        def draw_last_point(traj, color, radius=5, width=0):
            if traj is None or len(traj) == 0:
                return
            x, y = traj[-1]
            px, py = world_to_screen(float(x), float(y))
            pygame.draw.circle(self.screen, color, (px, py), radius, width)
        # Draw occupancy grid
        if self.occupancy_grid is not None:
            h, w = self.occupancy_grid.shape
            cell_screen_w = self.grid_resolution * scale
            cell_screen_h = self.grid_resolution * scale
            for i in range(h):
                for j in range(w):
                    if self.occupancy_grid[i, j] == 0:
                        continue
                    x_cell = (j - w / 2.0 + 0.5) * self.grid_resolution
                    y_cell = -(i - h / 2.0 + 0.5) * self.grid_resolution
                    px, py = world_to_screen(x_cell, y_cell)
                    cell_rect = pygame.Rect(
                        px - cell_screen_w / 2,
                        py - cell_screen_h / 2,
                        cell_screen_w,
                        cell_screen_h,
                    )
                    pygame.draw.rect(self.screen, obstacle_color, cell_rect)
        # Draw estimated occupancy overlay
        if self.visual_occupancy_grid is not None:
            vh, vw = self.visual_occupancy_grid.shape
            overlay_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
            cell_screen_w = self.grid_resolution * scale
            cell_screen_h = self.grid_resolution * scale
            v_grid = np.asarray(self.visual_occupancy_grid, dtype=np.float32)
            v_max = float(v_grid.max())
            if v_max > 1e-12:
                v_norm = v_grid / v_max
            else:
                v_norm = v_grid
            for i in range(vh):
                for j in range(vw):
                    intensity = float(v_norm[i, j])
                    if intensity < 1e-4:
                        continue
                    alpha = int(np.clip(intensity * 220, 15, 220))
                    color = (255, 140, 0, alpha)
                    x_cell = (j - vw / 2.0 + 0.5) * self.grid_resolution
                    y_cell = -(i - vh / 2.0 + 0.5) * self.grid_resolution
                    px, py = world_to_screen(x_cell, y_cell)
                    cell_rect = pygame.Rect(
                        px - cell_screen_w / 2,
                        py - cell_screen_h / 2,
                        cell_screen_w,
                        cell_screen_h,
                    )
                    pygame.draw.rect(overlay_surface, color, cell_rect)
            self.screen.blit(overlay_surface, (0, 0))
        # Draw distance map heatmap
        if self.distance_map is not None and self.occupancy_grid is not None:
            h, w = self.distance_map.shape
            finite_dists = self.distance_map[np.isfinite(self.distance_map)]
            max_dist = np.max(finite_dists) if finite_dists.size > 0 else 1.0
            heatmap_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
            for i in range(h):
                for j in range(w):
                    if self.occupancy_grid[i, j] == 0:
                        dist = self.distance_map[i, j]
                        if np.isfinite(dist):
                            ratio = dist / max_dist
                            color = (int(255 * ratio), 0, int(255 * (1 - ratio)), 100)
                        else:
                            color = (128, 128, 128, 100)
                        x_cell = (j - w / 2.0 + 0.5) * self.grid_resolution
                        y_cell = -(i - h / 2.0 + 0.5) * self.grid_resolution
                        px, py = world_to_screen(x_cell, y_cell)
                        cell_screen_w = self.grid_resolution * scale
                        cell_screen_h = self.grid_resolution * scale
                        rect = pygame.Rect(
                            px - cell_screen_w / 2,
                            py - cell_screen_h / 2,
                            cell_screen_w,
                            cell_screen_h
                        )
                        pygame.draw.rect(heatmap_surface, color, rect)
            self.screen.blit(heatmap_surface, (0, 0))
        # Draw goal
        gx, gy = self.goal_pos
        goal_px, goal_py = world_to_screen(gx, gy)
        pygame.draw.circle(self.screen, goal_color, (goal_px, goal_py), 7)
        # Draw trajectory overlays. If explicit overlays are not set, fall back to live trajectories.
        draw_polyline(self._overlay_real_traj, (255, 120, 120), width=2)
        draw_polyline(self._overlay_odom_traj if self._overlay_odom_traj is not None else self._odom_traj, (80, 180, 255), width=2)
        draw_polyline(self._overlay_true_traj if self._overlay_true_traj is not None else self._gt_traj, (120, 255, 120), width=2)
        draw_polyline(self._est_traj, (255, 0, 255), width=2)
        draw_last_point(self._overlay_real_traj, (255, 120, 120), radius=4)
        # True pose
        if gt_state is not None:
            x_true, y_true, heading_true, _ = gt_state
            true_px, true_py = world_to_screen(x_true, y_true)
            pygame.draw.circle(self.screen, true_color, (true_px, true_py), 6)
            tx2 = x_true + heading_len * np.sin(heading_true)
            ty2 = y_true + heading_len * np.cos(heading_true)
            tpx2, tpy2 = world_to_screen(tx2, ty2)
            pygame.draw.line(self.screen, true_color, (true_px, true_py), (tpx2, tpy2), 2)
        # Odom pose
        if odom_state is not None:
            x_o, y_o, heading_o, _ = odom_state
            odom_px, odom_py = world_to_screen(x_o, y_o)
            pygame.draw.circle(self.screen, odom_color, (odom_px, odom_py), 5)
            ox2 = x_o + heading_len * np.sin(heading_o)
            oy2 = y_o + heading_len * np.cos(heading_o)
            opx2, opy2 = world_to_screen(ox2, oy2)
            pygame.draw.line(self.screen, odom_color, (odom_px, odom_py), (opx2, opy2), 2)
        # Belief mean + covariance ellipse
        if self.belief_mean is not None and self.belief_cov is not None:
            mean = self.belief_mean
            cov = self.belief_cov
            if mean.shape[0] == 4:
                mean_pos = mean[0:2]
            else:
                mean_pos = mean
            if cov.shape == (4, 4):
                cov_pos = cov[0:2, 0:2]
            else:
                cov_pos = cov
            bx, by = float(mean_pos[0]), float(mean_pos[1])
            bpx, bpy = world_to_screen(bx, by)
            pygame.draw.circle(self.screen, belief_color, (bpx, bpy), 4, 1)
            self._draw_covariance_ellipse(self.screen, mean_pos, cov_pos, world_to_screen, belief_color, n_std=2.0)
        # Legend
        font = pygame.font.SysFont(None, 18)
        legend_lines = [
            "Green: ground truth",
            "Blue: odometry",
            "Magenta: estimate",
        ]
        for idx, text in enumerate(legend_lines):
            surf = font.render(text, True, (220, 220, 220))
            self.screen.blit(surf, (5, 5 + 18 * idx))
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        self.stop_logging()
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    def _draw_covariance_ellipse(self, surface, mean_pos, cov_pos, world_to_screen, color, n_std=2.0):
        cov_pos = 0.5 * (cov_pos + cov_pos.T)
        eigvals, eigvecs = np.linalg.eigh(cov_pos)
        eigvals = np.maximum(eigvals, 1e-12)
        r1 = n_std * math.sqrt(eigvals[0])
        r2 = n_std * math.sqrt(eigvals[1])
        order = np.argsort(eigvals)
        v = eigvecs[:, order[1]]
        angle = math.atan2(v[1], v[0])
        points = []
        for t in np.linspace(0, 2 * math.pi, num=40, endpoint=True):
            x_local = r1 * math.cos(t)
            y_local = r2 * math.sin(t)
            x_rot = x_local * math.cos(angle) - y_local * math.sin(angle)
            y_rot = x_local * math.sin(angle) + y_local * math.cos(angle)
            x_world = mean_pos[0] + x_rot
            y_world = mean_pos[1] + y_rot
            px, py = world_to_screen(x_world, y_world)
            points.append((px, py))
        if len(points) >= 2:
            pygame.draw.polygon(surface, color, points, width=1)

# LabVisualiser wrapper for labs
class LabVisualiser:
    def __init__(self, sim_env, real_env=None, csv_path=None):
        self.sim_env = sim_env
        self.real_env = real_env
        self.real_mode = real_env is not None
        self.csv_path = csv_path

        if self.csv_path is not None and hasattr(self.sim_env, "vis"):
            self.sim_env.vis.csv_path = self.csv_path

        # Expose model params
        for attr in ["dt", "vel_limit", "max_turn_rate", "heading_tolerance", "speed_scale", "goal_pos"]:
            setattr(self, attr, getattr(sim_env, attr, None))
    def start_logging(self):
        if self.csv_path is not None:
            self.sim_env.vis.csv_path = self.csv_path
        self.sim_env.vis.start_logging()
    def stop_logging(self):
        self.sim_env.vis.stop_logging()
    def reset(self):
        self.sim_env.vis.reset()
        if self.real_mode:
            self.sim_env.reset()
    def update(self, action, obs, info, ekf_mean=None, ekf_cov=None):
        # In dual-env setups, mirror action to the render env.
        if self.real_mode and self.real_env is not None and self.sim_env is not self.real_env:
            self.sim_env.step(action)
            gt = self.sim_env.state_true
        else:
            gt = info.get("state_true")
        odom = info.get("state_odom")
        setpoint = info.get("setpoint_xy") if isinstance(info, dict) else None
        self.sim_env.vis.record(gt, odom, action, ekf_mean, ekf_cov, setpoint=setpoint)
        if ekf_mean is not None and ekf_cov is not None:
            self.sim_env.vis.set_belief(ekf_mean, ekf_cov)
        # Overlay trajectories
        sim_odom_traj = self.sim_env.vis._odom_traj
        sim_gt_traj = self.sim_env.vis._gt_traj

        real_traj = None
        if self.real_mode:
            real_x, real_y, _ = self.real_env.get_live_traj()
            if len(real_x) > 0:
                real_traj = np.column_stack([np.array(real_x), np.array(real_y)])

        self.sim_env.vis.set_overlay_trajectories(real=real_traj, odom=sim_odom_traj, true=sim_gt_traj)

    def step(self, action, obs, info, ekf_mean=None, ekf_cov=None):
        """Backward-compatible helper: update visualiser state then render via env API."""
        self.update(action, obs, info, ekf_mean=ekf_mean, ekf_cov=ekf_cov)
        # Route rendering through the environment API (Gym-style) rather than calling visualiser directly.
        self.sim_env.render()
