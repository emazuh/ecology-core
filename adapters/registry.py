from .chains import (
    ChainSequential,
    ChainSequentialInputDependent,
    ChainParallelFixed,
    ChainParallelInputDependent,
)

CHAIN_MAP = {
    "seq": ChainSequential,
    "seq_input": ChainSequentialInputDependent,
    "par_fixed": ChainParallelFixed,
    "par_input": ChainParallelInputDependent,
}
