"""Runtime template bindings."""
from .prompt_loader import load_prompt

FALLBACK_JUDGE_PROMPT = load_prompt("phys_utils.prompts.FALLBACK_JUDGE_PROMPT")
TYPE_CLASSIFICATION_PROMPT_TEMPLATE = load_prompt("phys_utils.prompts.TYPE_CLASSIFICATION_PROMPT_TEMPLATE")
EXECUTE_MODELING_PROMPT = load_prompt("phys_utils.prompts.EXECUTE_MODELING_PROMPT")
