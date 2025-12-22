import os
from pathlib import Path

# 基础目录
BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "papers"
IMAGES_DIR = BASE_DIR / "images"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 索引文件路径
PAPER_INDEX_PATH = DATA_DIR / "paper_index.json"
CHUNK_INDEX_PATH = DATA_DIR / "chunk_index.json"
IMAGE_INDEX_PATH = DATA_DIR / "image_index.json"

# 模型与端口配置
# - TEXT_*：文本端口，必须支持 chat/completions（主题分类）。示例：http://HOST:8789/v1。
# - TEXT_EMBED_*：可选的独立 embedding 端口（例如部署 qwen_emb 在 8791），若不设则默认走 TEXT_BASE_URL。
# - VISION_*：多模态端口，用于图片描述/图文对齐（如 llava）。示例：http://HOST:8790/v1。
TEXT_BASE_URL = os.environ.get("TEXT_BASE_URL", "http://172.16.206.198:8789/v1")
TEXT_EMBED_BASE_URL = os.environ.get("TEXT_EMBED_BASE_URL", os.environ.get("TEXT_BASE_URL", "http://172.16.206.198:8791/v1"))
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "http://172.16.206.198:8790/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

# 后端暴露的模型名
TEXT_MODEL = os.environ.get("TEXT_MODEL", "qwen")
TEXT_EMBED_MODEL = os.environ.get("TEXT_EMBED_MODEL", "qwen_emb")
VISION_MODEL = os.environ.get("VISION_MODEL", "llava")

# 若文本端口没有 /embeddings，请置 0/false，直接跳过远程 embedding 调用，使用本地哈希向量。
PREFER_REMOTE_EMBEDDING = os.environ.get("PREFER_REMOTE_EMBEDDING", "1").lower() not in {
    "0",
    "false",
}

DEFAULT_TOP_K = 5
CHUNK_SIZE = 800  # 分页拆分时的每段字符数
MAX_CHUNKS_PER_DOC = 200
