import logging
import re

def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def compress_prompt(prompt, max_length=1024, max_words=50):
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    if len(prompt) <= max_length:
        return prompt
    prompt = prompt[:max_length]
    words = prompt.split()
    if len(words) > max_words:
        compressed_words = words[:10] + words[-10:]
        prompt = ' '.join(compressed_words)
    return prompt
