from enum import Enum


class TaskDifficulty(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class SubTaskDifficulty(str, Enum):
    EASY = "easy"
    HARD = "hard"


class NodeName(str, Enum):
    FIRST_LEVEL_ASSESSMENT = "first_level_assessment"
    DIRECT_ANSWER = "direct_answer"
    DECOMPOSER = "decomposer"
    SCHEDULER = "scheduler"
    SYNTHESIZER = "synthesizer"
