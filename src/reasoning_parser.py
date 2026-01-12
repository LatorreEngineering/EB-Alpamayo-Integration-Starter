"""
Reasoning Trace Parser

Utilities for extracting, parsing, and analyzing chain-of-thought and
chain-of-causation reasoning traces from Alpamayo 1 VLA model outputs.

This module provides:
- Structured parsing of reasoning steps
- Confidence and safety flag extraction
- Temporal reasoning analysis
- Visualization support

Example:
    >>> from src.reasoning_parser import ReasoningParser
    >>> parser = ReasoningParser()
    >>> steps = parser.extract_reasoning_steps(model_output.reasoning_trace)
    >>> for step in steps:
    ...     print(f"Step {step.id}: {step.thought}")
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import logging

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning steps in VLA outputs."""

    PERCEPTION = "perception"  # Scene understanding
    PREDICTION = "prediction"  # Future state prediction
    PLANNING = "planning"  # Decision making
    SAFETY = "safety"  # Safety assessment
    CAUSAL = "causal"  # Causal relationship
    META = "meta"  # Meta-reasoning about the task


@dataclass
class ReasoningStep:
    """
    Single step in the chain-of-thought reasoning process.

    Attributes:
        id: Step identifier (e.g., 1, 2, 3, ...)
        thought: The reasoning text
        reasoning_type: Type of reasoning (perception, planning, etc.)
        confidence: Confidence score (0-1), if available
        safety_flags: Safety-related concerns mentioned
        referenced_objects: Objects/entities mentioned in this step
        causal_links: IDs of steps this step depends on
        metadata: Additional parsed information
    """

    id: int
    thought: str
    reasoning_type: ReasoningType = ReasoningType.META
    confidence: Optional[float] = None
    safety_flags: Set[str] = field(default_factory=set)
    referenced_objects: List[str] = field(default_factory=list)
    causal_links: List[int] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """String representation."""
        return f"ReasoningStep({self.id}: {self.reasoning_type.value}, {self.thought[:50]}...)"


@dataclass
class ReasoningTrace:
    """
    Complete reasoning trace with all steps and analysis.

    Attributes:
        steps: List of reasoning steps
        total_steps: Total number of steps
        safety_critical: Whether any safety flags were raised
        confidence_avg: Average confidence across steps
        reasoning_graph: Dependency graph between steps
    """

    steps: List[ReasoningStep]
    total_steps: int
    safety_critical: bool
    confidence_avg: Optional[float] = None
    reasoning_graph: Dict[int, List[int]] = field(default_factory=dict)

    def get_steps_by_type(self, reasoning_type: ReasoningType) -> List[ReasoningStep]:
        """Filter steps by reasoning type."""
        return [s for s in self.steps if s.reasoning_type == reasoning_type]

    def get_safety_concerns(self) -> List[str]:
        """Extract all safety flags across all steps."""
        concerns = []
        for step in self.steps:
            concerns.extend(step.safety_flags)
        return list(set(concerns))  # Deduplicate


class ReasoningParser:
    """
    Parser for extracting structured reasoning from VLA model outputs.

    Handles different reasoning formats and extracts semantic information
    about the model's decision-making process.
    """

    # Common safety-related keywords
    SAFETY_KEYWORDS = {
        "collision",
        "crash",
        "danger",
        "unsafe",
        "risk",
        "pedestrian",
        "obstacle",
        "emergency",
        "brake",
        "stop",
        "caution",
        "warning",
        "hazard",
    }

    # Patterns for parsing different reasoning formats
    STEP_PATTERNS = [
        r"Step\s+(\d+):\s*(.+?)(?=Step\s+\d+:|$)",  # "Step N: ..."
        r"(\d+)\.\s*(.+?)(?=\d+\.|$)",  # "1. ..."
        r"\[Step\s+(\d+)\]\s*(.+?)(?=\[Step|$)",  # "[Step N] ..."
        r"Thought\s+(\d+):\s*(.+?)(?=Thought\s+\d+:|$)",  # "Thought N: ..."
    ]

    def __init__(self, safety_keywords: Optional[Set[str]] = None) -> None:
        """
        Initialize reasoning parser.

        Args:
            safety_keywords: Custom set of safety-related keywords to detect
        """
        self.safety_keywords = safety_keywords or self.SAFETY_KEYWORDS

    def extract_reasoning_steps(
        self, reasoning_text: str, format_hint: Optional[str] = None
    ) -> ReasoningTrace:
        """
        Extract structured reasoning steps from raw text.

        Args:
            reasoning_text: Raw reasoning text from model output
            format_hint: Optional hint about the text format

        Returns:
            ReasoningTrace with parsed steps and analysis

        Example:
            >>> text = '''
            ... Step 1: I observe a pedestrian crossing ahead.
            ... Step 2: I predict they will continue crossing.
            ... Step 3: I plan to slow down and yield.
            ... '''
            >>> trace = parser.extract_reasoning_steps(text)
            >>> print(f"Found {trace.total_steps} reasoning steps")
        """
        steps = self._parse_steps(reasoning_text)

        if not steps:
            logger.warning("No reasoning steps found in text")
            # Create single step with full text
            steps = [
                ReasoningStep(
                    id=1, thought=reasoning_text.strip(), reasoning_type=ReasoningType.META
                )
            ]

        # Analyze each step
        for step in steps:
            self._analyze_step(step)

        # Build reasoning graph
        reasoning_graph = self._build_reasoning_graph(steps)

        # Compute aggregate statistics
        safety_critical = any(len(s.safety_flags) > 0 for s in steps)
        confidences = [s.confidence for s in steps if s.confidence is not None]
        confidence_avg = sum(confidences) / len(confidences) if confidences else None

        trace = ReasoningTrace(
            steps=steps,
            total_steps=len(steps),
            safety_critical=safety_critical,
            confidence_avg=confidence_avg,
            reasoning_graph=reasoning_graph,
        )

        logger.info(
            f"Parsed {trace.total_steps} reasoning steps, "
            f"safety_critical={safety_critical}"
        )

        return trace

    def _parse_steps(self, text: str) -> List[ReasoningStep]:
        """
        Parse raw text into individual reasoning steps.

        Tries multiple regex patterns to handle different formats.
        """
        steps = []

        # Try each pattern
        for pattern in self.STEP_PATTERNS:
            matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
            found_steps = []

            for match in matches:
                step_id = int(match.group(1))
                thought = match.group(2).strip()

                if thought:  # Skip empty steps
                    found_steps.append(
                        ReasoningStep(
                            id=step_id, thought=thought, reasoning_type=ReasoningType.META
                        )
                    )

            if found_steps:
                steps = found_steps
                logger.debug(f"Matched pattern: {pattern[:30]}... ({len(steps)} steps)")
                break

        return steps

    def _analyze_step(self, step: ReasoningStep) -> None:
        """
        Analyze individual step to extract metadata.

        Modifies step in-place with:
        - Reasoning type classification
        - Confidence extraction
        - Safety flag detection
        - Referenced object extraction
        """
        thought_lower = step.thought.lower()

        # Classify reasoning type
        step.reasoning_type = self._classify_reasoning_type(thought_lower)

        # Extract confidence if present
        step.confidence = self._extract_confidence(step.thought)

        # Detect safety flags
        step.safety_flags = self._detect_safety_flags(thought_lower)

        # Extract referenced objects (basic NER)
        step.referenced_objects = self._extract_objects(step.thought)

    def _classify_reasoning_type(self, thought_lower: str) -> ReasoningType:
        """Classify the type of reasoning based on keywords."""
        # Simple keyword-based classification
        if any(
            word in thought_lower
            for word in ["observe", "see", "detect", "notice", "identify"]
        ):
            return ReasoningType.PERCEPTION

        if any(word in thought_lower for word in ["will", "predict", "expect", "likely"]):
            return ReasoningType.PREDICTION

        if any(word in thought_lower for word in ["plan", "decide", "choose", "should"]):
            return ReasoningType.PLANNING

        if any(word in thought_lower for word in ["safe", "danger", "risk", "collision"]):
            return ReasoningType.SAFETY

        if any(word in thought_lower for word in ["because", "therefore", "causes", "due to"]):
            return ReasoningType.CAUSAL

        return ReasoningType.META

    def _extract_confidence(self, thought: str) -> Optional[float]:
        """Extract confidence score if present in text."""
        # Look for patterns like "confidence: 0.85" or "(85%)"
        patterns = [
            r"confidence[:\s]+([0-9.]+)",
            r"\(([0-9]+)%\)",
            r"([0-9.]+)\s*confidence",
        ]

        for pattern in patterns:
            match = re.search(pattern, thought.lower())
            if match:
                value = float(match.group(1))
                # Normalize to 0-1 range
                if value > 1.0:
                    value = value / 100.0
                return min(1.0, max(0.0, value))

        return None

    def _detect_safety_flags(self, thought_lower: str) -> Set[str]:
        """Detect safety-related keywords in text."""
        flags = set()
        for keyword in self.safety_keywords:
            if keyword in thought_lower:
                flags.add(keyword)
        return flags

    def _extract_objects(self, thought: str) -> List[str]:
        """
        Extract mentioned objects/entities.

        Simple implementation using common automotive terms.
        Could be enhanced with proper NER models.
        """
        object_keywords = [
            "pedestrian",
            "vehicle",
            "car",
            "truck",
            "cyclist",
            "motorcycle",
            "traffic light",
            "stop sign",
            "lane",
            "road",
            "intersection",
            "crosswalk",
        ]

        found_objects = []
        thought_lower = thought.lower()

        for obj in object_keywords:
            if obj in thought_lower:
                found_objects.append(obj)

        return found_objects

    def _build_reasoning_graph(self, steps: List[ReasoningStep]) -> Dict[int, List[int]]:
        """
        Build dependency graph between reasoning steps.

        Looks for explicit references like "from step 2" or implicit causal links.

        Returns:
            Dictionary mapping step ID to list of step IDs it depends on
        """
        graph = {step.id: [] for step in steps}

        for step in steps:
            # Look for explicit step references
            pattern = r"(?:step|thought)\s+(\d+)"
            matches = re.finditer(pattern, step.thought.lower())

            for match in matches:
                referenced_id = int(match.group(1))
                if referenced_id in graph and referenced_id != step.id:
                    graph[step.id].append(referenced_id)

        return graph

    def format_trace_text(
        self, trace: ReasoningTrace, include_metadata: bool = True
    ) -> str:
        """
        Format reasoning trace as human-readable text.

        Args:
            trace: Parsed reasoning trace
            include_metadata: Whether to include step metadata

        Returns:
            Formatted text representation
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Reasoning Trace ({trace.total_steps} steps)")
        lines.append("=" * 80)

        if trace.safety_critical:
            lines.append("⚠️  SAFETY CONCERNS DETECTED")

        if trace.confidence_avg is not None:
            lines.append(f"Average Confidence: {trace.confidence_avg:.2%}")

        lines.append("")

        for step in trace.steps:
            lines.append(f"Step {step.id} [{step.reasoning_type.value.upper()}]")
            lines.append(f"  {step.thought}")

            if include_metadata:
                if step.confidence is not None:
                    lines.append(f"  Confidence: {step.confidence:.2%}")

                if step.safety_flags:
                    lines.append(f"  Safety Flags: {', '.join(step.safety_flags)}")

                if step.referenced_objects:
                    lines.append(f"  Objects: {', '.join(step.referenced_objects)}")

            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Example reasoning text
    example_text = """
    Step 1: I observe a pedestrian waiting at the crosswalk on the right side.
    Step 2: I predict the pedestrian will likely cross when the light changes (confidence: 0.9).
    Step 3: I detect a vehicle approaching from the left lane at high speed.
    Step 4: Due to the potential collision risk, I should slow down preemptively.
    Step 5: I plan to reduce speed to 15 mph and prepare to stop if necessary.
    """

    parser = ReasoningParser()
    trace = parser.extract_reasoning_steps(example_text)

    print(parser.format_trace_text(trace))

    print("\nSafety Concerns:")
    for concern in trace.get_safety_concerns():
        print(f"  - {concern}")

    print("\nPerception Steps:")
    for step in trace.get_steps_by_type(ReasoningType.PERCEPTION):
        print(f"  {step.id}: {step.thought[:60]}...")
