# methods/behavior_analysis/configs.py

import os

# Paths
DEFAULT_DATA_ROOT = "data"

# Models
DEFAULT_OPTIMIZER_MODEL = "gpt-5-mini"

# If residual probing uses the same local Qwen captioner pipeline,
# this can stay None. If later using OpenAI vision model for residuals, set a model here.
DEFAULT_RESIDUAL_MODEL = None

# Optimization
DEFAULT_STEPS = 20
DEFAULT_BATCH_SIZE = 24
DEFAULT_MAX_RESIDUAL_CASES = 5
DEFAULT_RESIDUALS_PER_CASE = 2

# Validation gate
DEFAULT_USE_VAL_GATE = True
DEFAULT_VAL_GATE_BATCH_SIZE = 24
DEFAULT_VAL_GATE_TOLERANCE = 0.0
DEFAULT_SKIP_FULL_VAL_IF_REJECTED = True

# Prompt selection
DEFAULT_INCLUDE_INIT_AS_CANDIDATE = True

# Prompts
BASELINE_PROMPT = "Write a caption for this image."

OBSERVER_SPECIFIC_SUPPORTED = (
    "Write one specific caption as a careful human observer would: "
    "state what is visibly happening, name the important subjects and objects, "
    "and include only scene meanings that are clearly supported."
)

PROMPT_UPDATE_CONSTRAINTS = [
    "The optimized text must be a prompt for generating one natural image caption.",
    "The captioning model must output exactly one caption, not tags, lists, multiple candidates, JSON, or a multi-line format.",
    "The prompt must not mention CLIP, SigLIP, scores, rerankers, verifiers, canonical IDs, residuals, or optimization.",
    "The prompt must not instruct the model to output fallback phrases such as 'image unclear'.",
    "The prompt must not be image-specific.",
]

OPENAI_API_KEY="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"
