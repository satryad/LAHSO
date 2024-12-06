from enum import Enum


class ExecutionStatus(Enum):
    NOT_STARTED = 0
    EXECUTING = 1
    PAUSED = 2
    FINISHED = 3
