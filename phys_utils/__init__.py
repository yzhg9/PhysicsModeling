"""Physics reasoning utilities with optional model dependencies loaded on demand."""
from importlib import import_module
_EXPORTS = {'JsonlWriter': 'io', 'load_records': 'io', 'load_existing_ids': 'io', 'load_existing_attempts': 'io', 'extract_boxed': 'parsing', 'extract_judge': 'parsing', 'split_blocks': 'parsing', 'run_async_map': 'runner', 'dedup_records': 'runner', 'build_todo': 'runner', 'split_many_value_unit': 'numerical', 'score_numerical': 'numerical', 'judge_numerical_answer': 'numerical', 'strip_latex_formatting': 'symbolic', 'score_symbolic': 'symbolic', 'judge_one': 'judge', 'AsyncLLMClient': 'llm_client', 'FALLBACK_JUDGE_PROMPT': 'prompts', 'EXECUTE_MODELING_PROMPT': 'prompts', 'TYPE_CLASSIFICATION_PROMPT_TEMPLATE': 'prompts'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(f'.{_EXPORTS[name]}', __name__), name)
    globals()[name] = value
    return value
