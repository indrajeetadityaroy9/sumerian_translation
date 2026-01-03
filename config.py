"""
Centralized configuration for Sumerian Translation Pipeline.

Supports both:
- LLM fine-tuning (Llama-3, primary)
- Legacy mT5 NMT (archived in legacy_mt5/)

All paths and configurations for training, evaluation, and inference.
"""

from pathlib import Path


# =============================================================================
# BASE PATHS
# =============================================================================

class Paths:
    """Centralized path configuration."""

    # Root directory
    ROOT = Path(__file__).parent

    # ==========================================================================
    # LLM TRAINING (Primary - Llama-3)
    # ==========================================================================

    # LLM model outputs
    MODELS_LLM = ROOT / "models_llm"

    # LLM-ready datasets (Alpaca format)
    FINAL_LLM_DATA = ROOT / "data" / "final_llm_ready"
    SFT_TRAIN = FINAL_LLM_DATA / "sft_train.json"
    SFT_TEST = FINAL_LLM_DATA / "sft_test.json"
    DPO_PAIRS = FINAL_LLM_DATA / "dpo_pairs.json"

    # LLM training configs
    CONFIGS_LLM = ROOT / "configs_llm"

    # ==========================================================================
    # RAW SOURCE DATA (For Evaluation & NE Metrics)
    # ==========================================================================

    RAW_SOURCE = ROOT / "data" / "raw_source"
    ETCSL_PARQUET = RAW_SOURCE / "etcsl_gold.parquet"
    GLOSSARY_PARQUET = RAW_SOURCE / "glossary_sux.parquet"
    ORACC_LITERARY_PARQUET = RAW_SOURCE / "oracc_literary.parquet"
    ORACC_ROYAL_PARQUET = RAW_SOURCE / "oracc_royal.parquet"

    # Archive with full metadata (for NE evaluation)
    ARCHIVE = ROOT / "data" / "archive"
    VALID_WITH_NE = ARCHIVE / "valid.jsonl"  # Has named_entities field

    # ==========================================================================
    # LEGACY PATHS (For backward compatibility with legacy_mt5/)
    # ==========================================================================

    # Legacy model checkpoints
    MODELS = ROOT / "models"
    NMT_CHECKPOINT = MODELS / "sumerian_mt5_final"

    # Legacy training data
    OUTPUT = ROOT / "output"
    TRAINING_DATA = ROOT / "output_training_v2_clean"
    TRAIN_FILE = TRAINING_DATA / "finetune" / "train_augmented.jsonl"
    TRAIN_FILE_V2 = TRAINING_DATA / "finetune" / "train_augmented_v2.jsonl"
    VALID_FILE = TRAINING_DATA / "finetune" / "valid.jsonl"

    # Legacy consolidated data (now in raw_source)
    CONSOLIDATED_DIR = RAW_SOURCE

    # Extracted corpora
    ETCSL_CORPUS = OUTPUT / "parallel_corpus.jsonl"

    # Graph Augmentation
    GRAPH_AUGMENTED = TRAINING_DATA / "finetune" / "train_graph_augmented.jsonl"
    AUDIT_DIR = TRAINING_DATA / "finetune" / "audit"

    # Ablation datasets (in archive)
    ABLATION_BASELINE = ARCHIVE / "train_baseline.jsonl"
    ABLATION_NAIVE = ARCHIVE / "train_exact_only.jsonl"
    ABLATION_SMART = ARCHIVE / "train_substitution.jsonl"

    @classmethod
    def ensure_dirs(cls):
        """Create all output directories if they don't exist."""
        cls.MODELS_LLM.mkdir(parents=True, exist_ok=True)
        cls.FINAL_LLM_DATA.mkdir(parents=True, exist_ok=True)
        cls.CONFIGS_LLM.mkdir(parents=True, exist_ok=True)
        cls.RAW_SOURCE.mkdir(parents=True, exist_ok=True)
        cls.ARCHIVE.mkdir(parents=True, exist_ok=True)
        cls.MODELS.mkdir(parents=True, exist_ok=True)
        cls.AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LLM CONFIGURATION (Primary)
# =============================================================================

class LLMConfig:
    """Configuration for Llama-3 fine-tuning."""

    # Base model
    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # LoRA hyperparameters
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Training hyperparameters
    MAX_SEQ_LEN = 512
    BATCH_SIZE = 4  # Per device (with gradient accumulation)
    GRAD_ACCUM = 16  # Effective batch = 4 * 16 * 2 GPUs = 128
    EPOCHS = 3
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.03
    WEIGHT_DECAY = 0.01

    # Precision
    USE_BF16 = True
    USE_FLASH_ATTN = True

    # Generation
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.1
    TOP_P = 0.9

    # Prompt template (Llama-3 Instruct format)
    SYSTEM_PROMPT = """You are an expert Sumerologist specializing in translating ancient Sumerian cuneiform texts into modern English. Provide accurate, scholarly translations."""

    @classmethod
    def format_prompt(cls, instruction: str, input_text: str) -> str:
        """Format a prompt in Llama-3 Instruct format."""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{cls.SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|}

{instruction}

{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    @classmethod
    def format_prompt_simple(cls, input_text: str) -> str:
        """Format a simple translation prompt."""
        return cls.format_prompt(
            "Translate this Sumerian text into English:",
            input_text
        )


# =============================================================================
# LEGACY CONFIGURATION (For mT5 - archived)
# =============================================================================

class LegacyMT5Config:
    """Legacy mT5 configuration (for reference and comparison)."""

    MT5_MODELS = {
        "small": "google/mt5-small",    # 300M params
        "base": "google/mt5-base",      # 580M params
        "large": "google/mt5-large",    # 1.2B params
    }

    TRAINING_DEFAULTS = {
        "batch_size": 64,
        "epochs": 20,
        "learning_rate": 3e-5,
        "max_length": 128,
        "warmup_ratio": 0.06,
        "task_prefix": "translate Sumerian to English: ",
    }


# =============================================================================
# CONTROL TOKENS (Data Provenance)
# =============================================================================

class ControlTokens:
    """Control tokens for data provenance tracking."""
    GOLD = "<gold>"      # Original ETCSL gold data
    SILVER = "<silver>"  # High-confidence matches (skeleton >= 95%)
    AUG = "<aug>"        # Entity substitution augmented
    GLOSS = "<gloss>"    # Glossary-based augmented


# =============================================================================
# EVALUATION TARGETS
# =============================================================================

class EvalTargets:
    """Expected evaluation metric ranges."""

    # Translation quality
    BLEU_MIN = 15.0
    BLEU_MAX = 25.0
    CHRF_MIN = 30.0
    CHRF_MAX = 40.0

    # Semantic similarity
    BERTSCORE_F1_MIN = 0.60

    # Named Entity accuracy (the "killer metric")
    NE_ACCURACY_BASELINE = 0.0   # mT5 baseline (expected)
    NE_ACCURACY_TARGET = 0.60    # LLM target


# =============================================================================
# BACKWARD COMPATIBILITY (Aliases for legacy scripts)
# =============================================================================

# These allow legacy_mt5/ scripts to import from config without modification
class ModelConfigs:
    """Alias for legacy scripts."""
    MT5_MODELS = LegacyMT5Config.MT5_MODELS


class TrainingDefaults:
    """Alias for legacy scripts."""
    MT5 = LegacyMT5Config.TRAINING_DEFAULTS


def get_train_file() -> Path:
    """Get the appropriate training file (legacy compatibility)."""
    if Paths.TRAIN_FILE_V2.exists():
        return Paths.TRAIN_FILE_V2
    return Paths.TRAIN_FILE


def get_model_checkpoint() -> Path:
    """Get the best available mT5 checkpoint (legacy compatibility)."""
    mt5_paths = [
        Paths.MODELS / "sumerian_mt5_final",
        Paths.MODELS / "sumerian_mt5_continued",
        Paths.MODELS / "sumerian_mt5",
    ]
    for path in mt5_paths:
        if path.exists():
            return path
    return Paths.NMT_CHECKPOINT


def get_llm_checkpoint() -> Path:
    """Get the best available LLM checkpoint."""
    llm_paths = [
        Paths.MODELS_LLM / "sumerian_llama3_sft",
        Paths.MODELS_LLM / "sumerian_llama3_dpo",
    ]
    for path in llm_paths:
        if path.exists():
            return path
    return Paths.MODELS_LLM / "sumerian_llama3_sft"
