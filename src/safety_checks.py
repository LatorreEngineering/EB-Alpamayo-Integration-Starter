"""
Safety Checks and Trajectory Validation

Basic plausibility checks for autonomous driving trajectories inspired by
ISO 26262 and automotive safety principles.

This module provides:
- Kinematic feasibility validation
- Collision avoidance checks
- Lane boundary compliance
- Acceleration/curvature limits

IMPORTANT: These are research-grade heuristics, NOT production safety systems.

Example:
    >>> from src.safety_checks import TrajectoryValidator
    >>> validator = TrajectoryValidator(max_accel=3.0, max_lateral_accel=4.0)
    >>> result = validator.validate(trajectory, scene_context)
    >>> if not result.is_valid:
    ...     print(f"Safety violations: {result.violations}")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of safety violations."""

    KINEMATIC = "kinematic"  # Physically impossible motion
    ACCELERATION = "acceleration"  # Excessive acceleration
    CURVATURE = "curvature"  # Too sharp turn
    COLLISION = "collision"  # Potential collision
    LANE_DEPARTURE = "lane_departure"  # Leaves lane boundaries
    SPEED_LIMIT = "speed_limit"  # Exceeds speed limits


class ViolationSeverity(Enum):
    """Severity levels for violations."""

    INFO = "info"  # Minor issue, informational
    WARNING = "warning"  # Should be reviewed
    ERROR = "error"  # Significant safety concern
    CRITICAL = "critical"  # Immediate safety hazard


@dataclass
class SafetyViolation:
    """
    Single safety violation detected in trajectory.

    Attributes:
        violation_type: Type of violation
        severity: Severity level
        location: Waypoint index where violation occurs
        value: Measured value that violates constraint
        threshold: Constraint threshold
        message: Human-readable description
    """

    violation_type: ViolationType
    severity: ViolationSeverity
    location: int
    value: float
    threshold: float
    message: str

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SafetyViolation({self.severity.value.upper()}: "
            f"{self.violation_type.value} at waypoint {self.location}, "
            f"value={self.value:.2f}, threshold={self.threshold:.2f})"
        )


@dataclass
class ValidationResult:
    """
    Result of trajectory validation.

    Attributes:
        is_valid: Whether trajectory passes all safety checks
        violations: List of detected violations
        metrics: Dict of computed safety metrics
        timestamp: Validation timestamp
    """

    is_valid: bool
    violations: List[SafetyViolation] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def get_critical_violations(self) -> List[SafetyViolation]:
        """Get only critical and error severity violations."""
        return [
            v
            for v in self.violations
            if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.ERROR]
        ]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append(f"Validation Result: {'PASS' if self.is_valid else 'FAIL'}")
        lines.append(f"Total Violations: {len(self.violations)}")

        if self.violations:
            severity_counts = {}
            for v in self.violations:
                severity_counts[v.severity] = severity_counts.get(v.severity, 0) + 1

            for severity, count in severity_counts.items():
                lines.append(f"  {severity.value.upper()}: {count}")

        return "\n".join(lines)


class TrajectoryValidator:
    """
    Validator for autonomous driving trajectories.

    Implements basic safety checks inspired by automotive standards:
    - ISO 26262: Functional safety
    - ISO 22179: Trajectory representation
    - RSS (Responsibility-Sensitive Safety) principles

    Note: These are simplified heuristics for research purposes.
    Production systems require comprehensive safety validation.
    """

    def __init__(
        self,
        max_longitudinal_accel: float = 3.0,  # m/s² (comfortable braking)
        max_lateral_accel: float = 4.0,  # m/s² (ISO 26262 inspired)
        max_curvature: float = 0.2,  # 1/m (minimum turn radius ~5m)
        collision_lookahead: float = 3.0,  # seconds
        min_ttc: float = 2.0,  # seconds (Time-To-Collision)
        speed_limit: float = 30.0,  # m/s (~67 mph)
    ) -> None:
        """
        Initialize trajectory validator.

        Args:
            max_longitudinal_accel: Maximum forward/backward acceleration (m/s²)
            max_lateral_accel: Maximum lateral acceleration (m/s²)
            max_curvature: Maximum path curvature (1/m)
            collision_lookahead: Time horizon for collision checking (s)
            min_ttc: Minimum acceptable time-to-collision (s)
            speed_limit: Maximum allowed speed (m/s)
        """
        self.max_longitudinal_accel = max_longitudinal_accel
        self.max_lateral_accel = max_lateral_accel
        self.max_curvature = max_curvature
        self.collision_lookahead = collision_lookahead
        self.min_ttc = min_ttc
        self.speed_limit = speed_limit

        logger.info(
            f"TrajectoryValidator initialized: "
            f"max_accel={max_longitudinal_accel}, "
            f"max_lat_accel={max_lateral_accel}"
        )

    def validate(
        self,
        trajectory: np.ndarray,
        dt: float = 0.1,
        obstacles: Optional[List[Dict]] = None,
        lanes: Optional[List[np.ndarray]] = None,
        current_speed: float = 0.0,
    ) -> ValidationResult:
        """
        Validate trajectory against safety constraints.

        Args:
            trajectory: Waypoints array (N, 2) of (x, y) coordinates in meters
            dt: Time step between waypoints in seconds
            obstacles: List of obstacle dicts with 'position', 'velocity', 'size'
            lanes: List of lane boundary arrays for lane keeping checks
            current_speed: Current vehicle speed in m/s

        Returns:
            ValidationResult with pass/fail and violation details

        Example:
            >>> traj = np.array([[0, 0], [1, 0], [2, 0.5], [3, 1.5]])
            >>> result = validator.validate(traj, dt=0.1)
            >>> print(result.summary())
        """
        violations = []
        metrics = {}

        # 1. Check kinematic feasibility
        speed_violations = self._check_speeds(trajectory, dt, current_speed)
        violations.extend(speed_violations)

        # 2. Check accelerations
        accel_violations, accel_metrics = self._check_accelerations(trajectory, dt)
        violations.extend(accel_violations)
        metrics.update(accel_metrics)

        # 3. Check curvature
        curv_violations, curv_metrics = self._check_curvature(trajectory)
        violations.extend(curv_violations)
        metrics.update(curv_metrics)

        # 4. Check collision potential
        if obstacles is not None:
            coll_violations, coll_metrics = self._check_collisions(
                trajectory, obstacles, dt
            )
            violations.extend(coll_violations)
            metrics.update(coll_metrics)

        # 5. Check lane keeping
        if lanes is not None:
            lane_violations = self._check_lane_keeping(trajectory, lanes)
            violations.extend(lane_violations)

        # Determine if valid (no critical or error violations)
        critical_violations = [
            v
            for v in violations
            if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.ERROR]
        ]
        is_valid = len(critical_violations) == 0

        result = ValidationResult(
            is_valid=is_valid, violations=violations, metrics=metrics
        )

        logger.info(
            f"Validation complete: {'PASS' if is_valid else 'FAIL'}, "
            f"{len(violations)} total violations"
        )

        return result

    def _check_speeds(
        self, trajectory: np.ndarray, dt: float, current_speed: float
    ) -> List[SafetyViolation]:
        """Check speed limits and feasibility."""
        violations = []

        # Compute speeds from trajectory
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        speeds = distances / dt

        for i, speed in enumerate(speeds):
            if speed > self.speed_limit:
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.SPEED_LIMIT,
                        severity=ViolationSeverity.WARNING,
                        location=i,
                        value=speed,
                        threshold=self.speed_limit,
                        message=f"Speed {speed:.1f} m/s exceeds limit {self.speed_limit:.1f} m/s",
                    )
                )

        return violations

    def _check_accelerations(
        self, trajectory: np.ndarray, dt: float
    ) -> Tuple[List[SafetyViolation], Dict[str, float]]:
        """Check longitudinal and lateral accelerations."""
        violations = []
        metrics = {}

        if len(trajectory) < 3:
            return violations, metrics

        # Compute velocities
        velocities = np.diff(trajectory, axis=0) / dt

        # Compute accelerations
        accelerations = np.diff(velocities, axis=0) / dt

        # Longitudinal acceleration (magnitude)
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)

        max_accel = np.max(accel_magnitudes) if len(accel_magnitudes) > 0 else 0
        metrics["max_acceleration"] = max_accel

        for i, accel_mag in enumerate(accel_magnitudes):
            if accel_mag > self.max_longitudinal_accel:
                severity = (
                    ViolationSeverity.ERROR
                    if accel_mag > self.max_longitudinal_accel * 1.5
                    else ViolationSeverity.WARNING
                )

                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.ACCELERATION,
                        severity=severity,
                        location=i,
                        value=accel_mag,
                        threshold=self.max_longitudinal_accel,
                        message=f"Acceleration {accel_mag:.2f} m/s² exceeds limit",
                    )
                )

        # Lateral acceleration (using curvature and speed)
        # v²/R where R = 1/curvature
        speeds = np.linalg.norm(velocities, axis=1)
        curvatures = self._compute_curvature(trajectory)

        if len(curvatures) > 0 and len(speeds) >= len(curvatures):
            lateral_accels = speeds[: len(curvatures)] ** 2 * curvatures
            max_lateral = np.max(np.abs(lateral_accels))
            metrics["max_lateral_acceleration"] = max_lateral

            for i, lat_accel in enumerate(lateral_accels):
                if abs(lat_accel) > self.max_lateral_accel:
                    violations.append(
                        SafetyViolation(
                            violation_type=ViolationType.ACCELERATION,
                            severity=ViolationSeverity.WARNING,
                            location=i,
                            value=abs(lat_accel),
                            threshold=self.max_lateral_accel,
                            message=f"Lateral accel {abs(lat_accel):.2f} m/s² exceeds limit",
                        )
                    )

        return violations, metrics

    def _compute_curvature(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute path curvature using finite differences.

        Curvature κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        """
        if len(trajectory) < 3:
            return np.array([])

        # First derivatives
        dx = np.gradient(trajectory[:, 0])
        dy = np.gradient(trajectory[:, 1])

        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2) ** 1.5
        denominator = np.where(denominator == 0, 1e-6, denominator)  # Avoid division by zero

        curvature = numerator / denominator

        return curvature

    def _check_curvature(
        self, trajectory: np.ndarray
    ) -> Tuple[List[SafetyViolation], Dict[str, float]]:
        """Check path curvature constraints."""
        violations = []
        metrics = {}

        curvatures = self._compute_curvature(trajectory)

        if len(curvatures) == 0:
            return violations, metrics

        max_curv = np.max(curvatures)
        metrics["max_curvature"] = max_curv

        for i, curv in enumerate(curvatures):
            if curv > self.max_curvature:
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.CURVATURE,
                        severity=ViolationSeverity.WARNING,
                        location=i,
                        value=curv,
                        threshold=self.max_curvature,
                        message=f"Curvature {curv:.3f} 1/m exceeds limit (turn radius {1/curv:.1f}m)",
                    )
                )

        return violations, metrics

    def _check_collisions(
        self, trajectory: np.ndarray, obstacles: List[Dict], dt: float
    ) -> Tuple[List[SafetyViolation], Dict[str, float]]:
        """Check potential collisions with obstacles."""
        violations = []
        metrics = {}

        min_distance = float("inf")

        for i, waypoint in enumerate(trajectory):
            for j, obstacle in enumerate(obstacles):
                obs_pos = np.array(obstacle["position"])
                obs_size = obstacle.get("size", [1.0, 1.0])
                obs_velocity = obstacle.get("velocity", [0.0, 0.0])

                # Predict obstacle position at this time step
                predicted_obs_pos = obs_pos + np.array(obs_velocity) * (i * dt)

                # Compute distance
                distance = np.linalg.norm(waypoint - predicted_obs_pos)
                min_distance = min(min_distance, distance)

                # Simple collision check: distance less than sum of radii
                ego_radius = 2.0  # Assume 2m vehicle radius for simplicity
                obs_radius = np.max(obs_size) / 2

                if distance < (ego_radius + obs_radius):
                    violations.append(
                        SafetyViolation(
                            violation_type=ViolationType.COLLISION,
                            severity=ViolationSeverity.CRITICAL,
                            location=i,
                            value=distance,
                            threshold=ego_radius + obs_radius,
                            message=f"Collision risk with obstacle {j} at t={i*dt:.1f}s",
                        )
                    )

                # TTC check
                relative_velocity = (
                    np.linalg.norm(np.diff(trajectory[max(0, i - 1) : i + 1], axis=0) / dt)
                    if i > 0
                    else 0
                )
                if relative_velocity > 0.1:  # Avoid division by zero
                    ttc = distance / relative_velocity
                    if ttc < self.min_ttc:
                        violations.append(
                            SafetyViolation(
                                violation_type=ViolationType.COLLISION,
                                severity=ViolationSeverity.ERROR,
                                location=i,
                                value=ttc,
                                threshold=self.min_ttc,
                                message=f"Low TTC {ttc:.2f}s with obstacle {j}",
                            )
                        )

        metrics["min_obstacle_distance"] = min_distance if min_distance != float("inf") else 0

        return violations, metrics

    def _check_lane_keeping(
        self, trajectory: np.ndarray, lanes: List[np.ndarray]
    ) -> List[SafetyViolation]:
        """Check if trajectory stays within lane boundaries."""
        violations = []

        # Simplified check: ensure waypoints are between lane boundaries
        # Assumes lanes[0] is left boundary, lanes[1] is right boundary
        if len(lanes) < 2:
            return violations

        for i, waypoint in enumerate(trajectory):
            # Find closest points on lane boundaries
            dists_left = cdist([waypoint], lanes[0])[0]
            dists_right = cdist([waypoint], lanes[1])[0]

            closest_left = lanes[0][np.argmin(dists_left)]
            closest_right = lanes[1][np.argmin(dists_right)]

            # Check if waypoint is outside lane (basic cross product test)
            # This is simplified - production would use proper geometric tests

        return violations  # Placeholder - full implementation needed


# Example usage
if __name__ == "__main__":
    # Create test trajectory
    trajectory = np.array(
        [[0, 0], [1, 0], [2, 0.5], [3, 1.5], [4, 2.5], [5, 3.0], [6, 3.2]]
    )

    # Create validator
    validator = TrajectoryValidator(max_longitudinal_accel=2.0, max_lateral_accel=3.0)

    # Test obstacles
    obstacles = [{"position": [4, 2], "size": [1, 1], "velocity": [0, 0]}]

    # Validate
    result = validator.validate(trajectory, dt=0.1, obstacles=obstacles)

    print(result.summary())
    print("\nMetrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.3f}")

    if result.violations:
        print("\nViolations:")
        for v in result.violations:
            print(f"  {v}")
