"""
Utilities for handling emoji embeddings in the model.
"""

import torch
from transformers import AutoTokenizer
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# Default emoji to Vietnamese word mapping
DEFAULT_EMOJI_MAPPING = {
    "😊": "vui",
    "😃": "vui",
    "😄": "vui",
    "😁": "vui",
    "😆": "vui",
    "😂": "vui",
    "🤣": "vui",
    "😢": "buồn",
    "😭": "buồn",
    "😔": "buồn",
    "😞": "buồn",
    "😡": "giận",
    "😠": "giận",
    "🤬": "giận",
    "😱": "sợ",
    "😨": "sợ",
    "😰": "sợ",
    "🤢": "ghê_tởm",
    "🤮": "ghê_tởm",
    "😲": "ngạc_nhiên",
    "😮": "ngạc_nhiên",
    "😯": "ngạc_nhiên",
    "❤️": "yêu",
    "💔": "đau",
    "👍": "tốt",
    "👎": "xấu",
}


def apply_emoji_embeddings(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    emoji_mapping: Optional[Dict[str, str]] = None
) -> torch.nn.Module:
    """
    Apply emoji embeddings to the model by copying embeddings from Vietnamese words.
    
    This function:
    1. Adds emojis to the tokenizer vocabulary if not present
    2. Resizes model embeddings to accommodate new tokens
    3. Copies embedding weights from corresponding Vietnamese words to emojis
    
    Args:
        model: The PhoBERT model
        tokenizer: The tokenizer
        emoji_mapping: Dictionary mapping emojis to Vietnamese words
    
    Returns:
        Modified model with emoji embeddings
    """
    if emoji_mapping is None:
        emoji_mapping = DEFAULT_EMOJI_MAPPING
    
    logger.info(f"Applying emoji embeddings for {len(emoji_mapping)} emojis")
    
    # Get current vocabulary size
    original_vocab_size = len(tokenizer)
    
    # Add emojis to tokenizer if not present
    emojis_to_add = []
    for emoji in emoji_mapping.keys():
        if emoji not in tokenizer.get_vocab():
            emojis_to_add.append(emoji)
    
    if emojis_to_add:
        logger.info(f"Adding {len(emojis_to_add)} new emoji tokens to tokenizer")
        tokenizer.add_tokens(emojis_to_add)
        
        # Resize model embeddings
        new_vocab_size = len(tokenizer)
        model.resize_token_embeddings(new_vocab_size)
        logger.info(f"Resized embeddings from {original_vocab_size} to {new_vocab_size}")
    
    # Get embedding layer
    embeddings = model.get_embeddings()
    
    # Copy embeddings from Vietnamese words to emojis
    with torch.no_grad():
        for emoji, vietnamese_word in emoji_mapping.items():
            emoji_id = tokenizer.convert_tokens_to_ids(emoji)
            
            # Tokenize the Vietnamese word
            word_tokens = tokenizer.tokenize(vietnamese_word)
            
            if not word_tokens:
                logger.warning(f"Could not tokenize Vietnamese word '{vietnamese_word}' for emoji '{emoji}'")
                continue
            
            # Get the ID of the first token (or average if multiple tokens)
            word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            
            if len(word_ids) == 1:
                # Single token: direct copy
                word_embedding = embeddings.weight[word_ids[0]]
            else:
                # Multiple tokens: average their embeddings
                word_embeddings = torch.stack([embeddings.weight[wid] for wid in word_ids])
                word_embedding = word_embeddings.mean(dim=0)
            
            # Copy to emoji embedding
            embeddings.weight[emoji_id] = word_embedding
            logger.debug(f"Copied embedding from '{vietnamese_word}' to '{emoji}'")
    
    logger.info("Emoji embeddings applied successfully")
    return model


def load_emoji_mapping_from_file(filepath: str) -> Dict[str, str]:
    """
    Load emoji mapping from a text file.
    
    File format (one per line):
    emoji,vietnamese_word
    
    Args:
        filepath: Path to the mapping file
    
    Returns:
        Dictionary mapping emojis to Vietnamese words
    """
    mapping = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) == 2:
                    emoji, word = parts
                    mapping[emoji.strip()] = word.strip()
    
    logger.info(f"Loaded {len(mapping)} emoji mappings from {filepath}")
    return mapping
