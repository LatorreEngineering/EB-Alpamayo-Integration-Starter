"""
Trajectory Visualization

Utilities for visualizing predicted trajectories and driving scenarios.
Supports bird's-eye view, overlays on camera images, and interactive plots.

Example:
    >>> from src.trajectory_viz import TrajectoryVisualizer
    >>> viz = TrajectoryVisualizer()
    >>> fig = viz.plot_birdseye(
    ...     trajectory=predicted_waypoints,
    ...     obstacles=detected_objects
    ... )
    >>> fig.show()
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.collections import LineCollection

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """
    Visualization toolkit for autonomous driving trajectories and scenarios.

    Provides methods for:
    - Bird's-eye view trajectory plots
    - Multi-trajectory comparison
    - Obstacle and lane visualization
    - Confidence interval rendering
    """

    # Color scheme (professional automotive palette)
    COLORS = {
        "ego": "#2E86AB",  # Blue
        "predicted": "#A23B72",  # Purple
        "reference": "#F18F01",  # Orange
        "obstacle": "#C73E1D",  # Red
        "lane": "#6C757D",  # Gray
        "safe_zone": "#06A77D",  # Green
    }

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100) -> None:
        """
        Initialize visualizer.

        Args:
            figsize: Default figure size in inches (width, height)
            dpi: Dots per inch for figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi

        # Set matplotlib style
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_birdseye(
        self,
        trajectory: np.ndarray,
        ego_position: Optional[np.ndarray] = None,
        obstacles: Optional[List[Dict]] = None,
        lanes: Optional[List[np.ndarray]] = None,
        reference_trajectory: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        title: str = "Trajectory Visualization",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create bird's-eye view visualization of trajectory and scene.

        Args:
            trajectory: Predicted waypoints (N, 2) array of (x, y) in meters
            ego_position: Current vehicle position (x, y, heading) in meters/radians
            obstacles: List of obstacle dicts with 'position', 'size', 'type'
            lanes: List of lane boundary arrays (N, 2)
            reference_trajectory: Ground truth/reference trajectory for comparison
            confidence: Confidence values (N,) for each waypoint
            title: Plot title
            xlim: X-axis limits (min, max)
            ylim: Y-axis limits (min, max)
            save_path: If provided, save figure to this path

        Returns:
            Matplotlib figure object

        Example:
            >>> trajectory = np.array([[0, 0], [1, 0.1], [2, 0.2], [3, 0.1]])
            >>> ego_pos = np.array([0, 0, 0])  # x, y, heading
            >>> obstacles = [
            ...     {'position': [5, 2], 'size': [2, 1], 'type': 'vehicle'},
            ...     {'position': [3, -1], 'size': [0.5, 0.5], 'type': 'pedestrian'}
            ... ]
            >>> fig = viz.plot_birdseye(trajectory, ego_pos, obstacles)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot lanes if provided
        if lanes is not None:
            for lane in lanes:
                ax.plot(
                    lane[:, 0],
                    lane[:, 1],
                    color=self.COLORS["lane"],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.6,
                    label="Lane boundary" if lane is lanes[0] else "",
                )

        # Plot reference trajectory if provided
        if reference_trajectory is not None:
            ax.plot(
                reference_trajectory[:, 0],
                reference_trajectory[:, 1],
                color=self.COLORS["reference"],
                linestyle=":",
                linewidth=2,
                marker="o",
                markersize=4,
                alpha=0.7,
                label="Reference",
            )

        # Plot predicted trajectory with confidence coloring
        if confidence is not None:
            # Create line segments with varying colors based on confidence
            points = trajectory.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(
                segments,
                cmap="viridis",
                linewidths=3,
                alpha=0.8,
            )
            lc.set_array(confidence[:-1])
            lc.set_clim(0, 1)
            line = ax.add_collection(lc)

            # Add colorbar
            cbar = fig.colorbar(line, ax=ax)
            cbar.set_label("Confidence", rotation=270, labelpad=20)

        else:
            # Simple line plot
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=self.COLORS["predicted"],
                linewidth=3,
                marker="o",
                markersize=6,
                label="Predicted trajectory",
            )

        # Plot waypoint markers
        ax.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            color=self.COLORS["predicted"],
            s=50,
            zorder=5,
            edgecolors="white",
            linewidths=1,
        )

        # Plot ego vehicle
        if ego_position is not None:
            self._plot_vehicle(
                ax,
                position=ego_position[:2],
                heading=ego_position[2] if len(ego_position) > 2 else 0,
                color=self.COLORS["ego"],
                label="Ego vehicle",
            )

        # Plot obstacles
        if obstacles is not None:
            for obs in obstacles:
                self._plot_obstacle(ax, obs)

        # Set axis properties
        ax.set_xlabel("X (meters)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y (meters)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # Set limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        # Legend
        ax.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")

        return fig

    def _plot_vehicle(
        self,
        ax: plt.Axes,
        position: np.ndarray,
        heading: float,
        color: str,
        label: str = "",
        length: float = 4.5,
        width: float = 2.0,
    ) -> None:
        """
        Plot vehicle as oriented rectangle.

        Args:
            ax: Matplotlib axes
            position: Vehicle position (x, y)
            heading: Vehicle heading in radians
            color: Fill color
            label: Legend label
            length: Vehicle length in meters
            width: Vehicle width in meters
        """
        # Create rectangle centered at origin
        rect_points = np.array(
            [
                [-length / 2, -width / 2],
                [length / 2, -width / 2],
                [length / 2, width / 2],
                [-length / 2, width / 2],
            ]
        )

        # Rotation matrix
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        # Rotate and translate
        rotated_points = rect_points @ rot_matrix.T + position

        # Create polygon
        vehicle = Polygon(
            rotated_points, closed=True, color=color, alpha=0.7, label=label, zorder=10
        )
        ax.add_patch(vehicle)

        # Add heading indicator (arrow)
        arrow_start = position
        arrow_end = position + length / 2 * np.array([np.cos(heading), np.sin(heading)])
        ax.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            arrowprops=dict(arrowstyle="->", color="white", lw=2),
            zorder=11,
        )

    def _plot_obstacle(self, ax: plt.Axes, obstacle: Dict) -> None:
        """
        Plot obstacle on axes.

        Args:
            ax: Matplotlib axes
            obstacle: Dict with 'position', 'size', 'type' keys
        """
        position = np.array(obstacle["position"])
        size = obstacle.get("size", [1.0, 1.0])
        obs_type = obstacle.get("type", "unknown")

        # Color based on type
        if obs_type == "pedestrian":
            color = "#FF6B6B"
            shape = "circle"
        elif obs_type == "vehicle":
            color = self.COLORS["obstacle"]
            shape = "rectangle"
        else:
            color = "#95A5A6"
            shape = "rectangle"

        # Draw shape
        if shape == "circle":
            circle = Circle(
                position,
                radius=max(size) / 2,
                color=color,
                alpha=0.6,
                label=f"Obstacle ({obs_type})",
                zorder=8,
            )
            ax.add_patch(circle)
        else:
            rect = Rectangle(
                position - np.array(size) / 2,
                size[0],
                size[1],
                color=color,
                alpha=0.6,
                label=f"Obstacle ({obs_type})",
                zorder=8,
            )
            ax.add_patch(rect)

    def plot_trajectory_comparison(
        self,
        trajectories: Dict[str, np.ndarray],
        title: str = "Trajectory Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare multiple trajectories in one plot.

        Args:
            trajectories: Dict mapping labels to trajectory arrays (N, 2)
            title: Plot title
            save_path: If provided, save figure to this path

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

        for (label, trajectory), color in zip(trajectories.items(), colors):
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                label=label,
                linewidth=2,
                marker="o",
                markersize=4,
                color=color,
            )

        ax.set_xlabel("X (meters)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y (meters)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")

        return fig

    def plot_metrics_over_time(
        self,
        time: np.ndarray,
        metrics: Dict[str, np.ndarray],
        title: str = "Trajectory Metrics",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot trajectory metrics over time.

        Args:
            time: Time array (N,)
            metrics: Dict mapping metric names to value arrays (N,)
            title: Plot title
            ylabel: Y-axis label
            save_path: If provided, save figure to this path

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for label, values in metrics.items():
            ax.plot(time, values, label=label, linewidth=2, marker="o", markersize=4)

        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")

        return fig


# Example usage
if __name__ == "__main__":
    # Create sample trajectory
    t = np.linspace(0, 10, 20)
    trajectory = np.column_stack([t, np.sin(t * 0.5) * 2])

    # Create visualizer
    viz = TrajectoryVisualizer()

    # Example 1: Basic bird's-eye view
    ego_pos = np.array([0, 0, 0])
    obstacles = [
        {"position": [7, 1.5], "size": [2, 1], "type": "vehicle"},
        {"position": [5, -1], "size": [0.5, 0.5], "type": "pedestrian"},
    ]

    fig = viz.plot_birdseye(
        trajectory=trajectory,
        ego_position=ego_pos,
        obstacles=obstacles,
        title="Sample Trajectory Visualization",
    )
    plt.show()

    # Example 2: Trajectory comparison
    trajectories = {
        "Predicted": trajectory,
        "Conservative": trajectory - np.array([0, 0.5]),
        "Aggressive": trajectory + np.array([0, 0.5]),
    }

    fig = viz.plot_trajectory_comparison(trajectories, title="Planning Variants")
    plt.show()
