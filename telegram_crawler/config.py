"""
Configuration file for Telegram Crawler
"""
import os
from pathlib import Path
from dotenv import load_dotenv

try:
    import cfg as legacy_cfg
except ImportError:
    legacy_cfg = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Chỉ load file .env thật ở root project.
# Không load .env.example để tránh dùng nhầm giá trị mẫu vào runtime.
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)


def _legacy_value(name, default=None):
    if legacy_cfg is None:
        return default
    return getattr(legacy_cfg, name, default)

# ==================== TELEGRAM CONFIG ====================
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID", str(_legacy_value("api_id", "")))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", str(_legacy_value("api_hash", "")))
TELEGRAM_SESSION = os.getenv("TELEGRAM_SESSION", "session")

# ==================== MONGODB CONFIG ====================
MONGO_URI = os.getenv("MONGO_URI", _legacy_value("MONGO_URI", "mongodb://localhost:27017/"))
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", _legacy_value("database", "telegram_data"))
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "messages")

# ==================== CRAWLER CONFIG ====================
# Danh sách các channels cần crawl
_channels_env = os.getenv("CHANNELS_TO_CRAWL", "").strip()
if _channels_env:
    CHANNELS_TO_CRAWL = [item.strip() for item in _channels_env.split(",") if item.strip()]
else:
    CHANNELS_TO_CRAWL = list(_legacy_value("channels", []))

# Số lượng tin nhắn tối đa crawl từ mỗi channel (None = tất cả)
_crawl_limit_env = os.getenv("CRAWL_LIMIT", "").strip().lower()
CRAWL_LIMIT = None if not _crawl_limit_env or _crawl_limit_env == "none" else int(_crawl_limit_env)

# Chỉ crawl các tin nhắn trong ngày hôm nay theo timezone cấu hình
CRAWL_TODAY_ONLY = os.getenv("CRAWL_TODAY_ONLY", "True").lower() == "true"

# Timezone dùng để xác định "hôm nay"
CRAWL_TIMEZONE = os.getenv("CRAWL_TIMEZONE", "Asia/Ho_Chi_Minh")

# Delay giữa các channels (giây)
CHANNEL_DELAY = 2

# ==================== MEDIA CONFIG ====================
# Thư mục lưu media
MEDIA_DIR = os.getenv("MEDIA_DIR", "./media")

# Thư mục lưu dữ liệu local khi MongoDB không khả dụng
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "./data")

# Download media (True/False)
DOWNLOAD_MEDIA = os.getenv("DOWNLOAD_MEDIA", "True").lower() == "true"

# Timeout tải media theo giây. Đặt None hoặc 0 để tắt timeout.
_media_timeout_env = os.getenv("MEDIA_DOWNLOAD_TIMEOUT", "30").strip().lower()
MEDIA_DOWNLOAD_TIMEOUT = None if not _media_timeout_env or _media_timeout_env in {"none", "0"} else int(_media_timeout_env)

# Các loại media cần download (photo, video, file)
MEDIA_TYPES = ["photo", "video"]

# ==================== LOGGING CONFIG ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "crawler.log")

# ==================== DATABASE INDEXES ====================
# Tạo các index trong MongoDB
CREATE_INDEXES = True

# ==================== DEBUG ====================
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
