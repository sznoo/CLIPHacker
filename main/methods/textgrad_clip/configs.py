# methods/textgrad_clip/configs.py

DEFAULT_ENGINE = "experimental:gpt-5-mini"
DEFAULT_MULTIMODAL_FEEDBACK_ENGINE = "gpt-5-mini"
DEFAULT_DATA_ROOT = "data"
DEFAULT_STEPS = 20
DEFAULT_BATCH_SIZE = 24
DEFAULT_MAX_FEEDBACK_CASES = 5
DEFAULT_GRADIENT_MEMORY = 0

# current default
DEFAULT_FEEDBACK_MODE = "textual"  # choices: textual, multimodal

BASELINE_PROMPT = "Write a caption for this image."

OBSERVER_SPECIFIC_SUPPORTED = (
    "Write one specific caption as a careful human observer would: "
    "state what is visibly happening, name the important subjects and objects, "
    "and include only scene meanings that are clearly supported."
)

PROMPT_ROLE = (
    "A natural-language instruction prompt given to a frozen image captioning VLM. "
    "The prompt controls how the model captions diverse images."
)

TEXTGRAD_CONSTRAINTS = [
    "The optimized text must be a prompt for generating one natural image caption.",
    "The captioning model must output exactly one caption, not tags, lists, multiple candidates, JSON, or a multi-line format.",
    "The prompt must not mention CLIP, SigLIP, scores, rerankers, verifiers, canonical IDs, or optimization.",
    "The prompt must not instruct the model to output fallback phrases such as 'image unclear'.",
]

OPENAI_API_KEY="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"
