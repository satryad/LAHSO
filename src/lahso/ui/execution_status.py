from enum import Enum


class ExecutionStatus(Enum):
    """
    Execution Status. Reused to keep track of what we're doing for longer, pausable,
    processes.
    """
    NOT_STARTED = 0
    EXECUTING = 1
    PAUSED = 2
    FINISHED = 3
