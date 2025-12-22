import importlib
from functools import lru_cache

from . import config


def _load_openai():
    # 延迟加载，避免启动时大量模块导入导致卡顿
    return importlib.import_module("openai").OpenAI


@lru_cache(maxsize=1)
def get_text_client():
    OpenAI = _load_openai()
    return OpenAI(base_url=config.TEXT_BASE_URL, api_key=config.API_KEY)


@lru_cache(maxsize=1)
def get_embed_client():
    OpenAI = _load_openai()
    return OpenAI(base_url=config.TEXT_EMBED_BASE_URL, api_key=config.API_KEY)


@lru_cache(maxsize=1)
def get_vision_client():
    OpenAI = _load_openai()
    return OpenAI(base_url=config.VISION_BASE_URL, api_key=config.API_KEY)
