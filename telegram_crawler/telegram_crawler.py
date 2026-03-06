import asyncio
import argparse
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from pymongo import MongoClient
import logging
from config import (
    TELEGRAM_API_ID,
    TELEGRAM_API_HASH,
    TELEGRAM_SESSION,
    MONGO_URI,
    MONGO_DB_NAME,
    MONGO_COLLECTION_NAME,
    MEDIA_DIR,
    LOCAL_DATA_DIR,
    CHANNELS_TO_CRAWL,
    CRAWL_LIMIT,
    CRAWL_TIMEZONE,
    CRAWL_TODAY_ONLY,
    DOWNLOAD_MEDIA,
    MEDIA_DOWNLOAD_TIMEOUT,
    MEDIA_TYPES,
    LOG_LEVEL,
    LOG_FILE,
    CREATE_INDEXES,
)

# Cấu hình logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Tạo thư mục lưu media
if not os.path.exists(MEDIA_DIR):
    os.makedirs(MEDIA_DIR)
    logger.info(f"✓ Tạo thư mục: {MEDIA_DIR}")

if not os.path.exists(LOCAL_DATA_DIR):
    os.makedirs(LOCAL_DATA_DIR)
    logger.info(f"✓ Tạo thư mục dữ liệu local: {LOCAL_DATA_DIR}")


class TelegramCrawler:
    def __init__(self):
        self.client = TelegramClient(TELEGRAM_SESSION, TELEGRAM_API_ID, TELEGRAM_API_HASH)
        self.mongo_client = None
        self.db = None
        self.collection = None
        self.session_stats = {}
        self.local_output_files = {}
        self.timezone = ZoneInfo(CRAWL_TIMEZONE)
        self.last_run_today_only = CRAWL_TODAY_ONLY
        self._init_mongo()

    def _init_mongo(self):
        """Kết nối MongoDB theo kiểu best-effort để crawler vẫn chạy được khi DB tắt."""
        try:
            self.mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            self.mongo_client.admin.command("ping")
            self.db = self.mongo_client[MONGO_DB_NAME]
            self.collection = self.db[MONGO_COLLECTION_NAME]

            if CREATE_INDEXES:
                self.collection.create_index("message_id")
                self.collection.create_index("channel_id")
                self.collection.create_index([("channel_id", 1), ("message_id", 1)], unique=True)
                logger.info("✓ Tạo indexes hoàn tất")

            logger.info("✓ Kết nối MongoDB thành công")
        except Exception as e:
            self.mongo_client = None
            self.db = None
            self.collection = None
            logger.warning(
                "⚠ Không kết nối được MongoDB (%s). Script sẽ tiếp tục chạy nhưng không lưu dữ liệu.",
                str(e),
            )

    def _get_local_output_path(self, channel_id, channel_username):
        safe_username = channel_username.replace("/", "_").replace("@", "")
        return os.path.join(LOCAL_DATA_DIR, f"channel_{channel_id}_{safe_username}.jsonl")

    def _save_to_local_file(self, channel_id, channel_username, message_data):
        """Lưu message ra file JSONL khi MongoDB không khả dụng."""
        output_path = self._get_local_output_path(channel_id, channel_username)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(message_data, ensure_ascii=False) + "\n")
        self.local_output_files[channel_id] = output_path

    def _message_local_date(self, message):
        if not message.date:
            return None
        return message.date.astimezone(self.timezone).date()

    async def download_media(self, message, channel_id, message_id):
        """
        Tải xuống media (ảnh/video) từ tin nhắn
        
        Returns:
            dict: Thông tin về media đã tải
        """
        if not DOWNLOAD_MEDIA:
            return None
            
        media_info = {
            "type": None,
            "path": None,
            "file_name": None
        }
        file_path = None
        
        try:
            if hasattr(message, 'media') and message.media:
                media = message.media
                
                # Tạo tên file dựa trên channel_id và message_id
                if isinstance(media, (MessageMediaPhoto, MessageMediaDocument)):
                    # Tạo thư mục cho channel
                    channel_media_dir = os.path.join(MEDIA_DIR, f"channel_{channel_id}")
                    if not os.path.exists(channel_media_dir):
                        os.makedirs(channel_media_dir)
                    
                    file_path = os.path.join(channel_media_dir, f"msg_{message_id}")
                    
                    # Xác định loại media
                    if isinstance(media, MessageMediaPhoto):
                        media_info["type"] = "photo"
                        file_path += ".jpg"
                    elif isinstance(media, MessageMediaDocument):
                        # Có thể là video, ảnh, hoặc file khác
                        mime_type = media.document.mime_type if media.document else "unknown"
                        if "video" in mime_type:
                            media_info["type"] = "video"
                            file_path += ".mp4"
                        elif "image" in mime_type:
                            media_info["type"] = "photo"
                            file_path += ".jpg"
                        else:
                            media_info["type"] = "file"
                            message_file = getattr(message, "file", None)
                            file_ext = getattr(message_file, "ext", None) or ".bin"
                            file_path += file_ext
                    
                    # Kiểm tra loại media có trong danh sách download không
                    if media_info["type"] not in MEDIA_TYPES:
                        return None
                    
                    # Tải file
                    logger.info(
                        "↳ Đang tải media cho message %s (%s) vào %s",
                        message_id,
                        media_info["type"],
                        file_path,
                    )
                    if MEDIA_DOWNLOAD_TIMEOUT is not None:
                        await asyncio.wait_for(
                            self.client.download_media(message, file=file_path),
                            timeout=MEDIA_DOWNLOAD_TIMEOUT,
                        )
                    else:
                        await self.client.download_media(message, file=file_path)
                    media_info["path"] = file_path
                    media_info["file_name"] = os.path.basename(file_path)
                    logger.info(f"✓ Đã tải media: {file_path}")
                    
        except asyncio.TimeoutError:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            logger.warning(
                "⚠ Timeout khi tải media từ message %s sau %s giây, bỏ qua file này",
                message_id,
                MEDIA_DOWNLOAD_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"⚠ Lỗi khi tải media từ message {message_id}: {str(e)}")
        
        return media_info if media_info["type"] else None

    async def extract_message_data(self, message, channel_id):
        """
        Trích xuất dữ liệu từ tin nhắn
        """
        # Tải media nếu có
        media_info = await self.download_media(message, channel_id, message.id)
        
        # Thông tin cơ bản
        message_data = {
            "message_id": message.id,
            "channel_id": channel_id,
            "text": message.text or "",
            "timestamp": message.date.isoformat() if message.date else None,
            "forward_from": None,
            "media": media_info,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Kiểm tra forward_from
        if message.forward:
            forward_info = message.forward
            if hasattr(forward_info, 'from_id') and forward_info.from_id:
                try:
                    # Lấy thông tin người gửi gốc
                    original_sender = await self.client.get_entity(forward_info.from_id)
                    message_data["forward_from"] = {
                        "user_id": forward_info.from_id.user_id if hasattr(forward_info.from_id, 'user_id') else None,
                        "username": original_sender.username if hasattr(original_sender, 'username') else None,
                        "first_name": original_sender.first_name if hasattr(original_sender, 'first_name') else None,
                    }
                except:
                    message_data["forward_from"] = {
                        "info": "Không thể lấy thông tin người gửi gốc"
                    }
        
        return message_data

    async def crawl_channel(self, channel_username, limit=None, today_only=CRAWL_TODAY_ONLY):
        """
        Crawl tin tức từ một channel
        
        Args:
            channel_username: tên channel (có hoặc không có @)
            limit: số lượng tin nhắn để crawl (None = tất cả)
            today_only: chỉ crawl tin của ngày hôm nay
        """
        try:
            # Làm sạch username
            if channel_username.startswith("@"):
                channel_username = channel_username[1:]
            
            logger.info(f"🔍 Đang crawl channel: {channel_username}")
            
            # Lấy thông tin channel
            try:
                channel = await self.client.get_entity(channel_username)
                channel_id = channel.id
                logger.info(f"✓ Channel ID: {channel_id}")
            except Exception as e:
                logger.error(f"✗ Không tìm thấy channel: {channel_username} - {str(e)}")
                return False
            
            # Crawl messages
            message_count = 0
            skipped_count = 0
            today = datetime.now(self.timezone).date()
            
            async for message in self.client.iter_messages(channel_username, limit=limit):
                try:
                    if today_only:
                        message_date = self._message_local_date(message)
                        if message_date is None:
                            continue
                        if message_date < today:
                            break
                        if message_date > today:
                            continue

                    # Kiểm tra xem message đã tồn tại trong DB chưa
                    if self.collection is not None:
                        existing = self.collection.find_one({
                            "channel_id": channel_id,
                            "message_id": message.id
                        })
                        
                        if existing:
                            skipped_count += 1
                            continue
                    
                    # Trích xuất dữ liệu
                    message_data = await self.extract_message_data(message, channel_id)
                    
                    # Lưu vào MongoDB
                    if self.collection is not None:
                        self.collection.insert_one(message_data)
                    else:
                        self._save_to_local_file(channel_id, channel_username, message_data)
                    message_count += 1
                    
                    if message_count % 10 == 0:
                        logger.info(f"↳ Đã xử lý {message_count} tin nhắn...")
                    
                except Exception as e:
                    logger.error(f"✗ Lỗi xử lý message {message.id}: {str(e)}")
                    continue

            self.session_stats[channel_id] = {
                "channel_username": channel_username,
                "new_messages": message_count,
                "skipped_messages": skipped_count,
                "output_path": self.local_output_files.get(channel_id),
                "today_only": today_only,
                "stored_in_db": self.collection is not None,
            }
            
            logger.info(f"✓ Crawl hoàn tất!")
            logger.info(f"  - Thêm mới: {message_count}")
            logger.info(f"  - Bỏ qua (đã tồn tại): {skipped_count}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Lỗi crawl channel: {str(e)}")
            return False

    async def crawl_multiple_channels(self, channels, limit=None, today_only=CRAWL_TODAY_ONLY):
        """
        Crawl từ nhiều channels
        
        Args:
            channels: danh sách tên channels
            limit: số lượng tin nhắn per channel
        """
        for channel in channels:
            await self.crawl_channel(channel, limit=limit, today_only=today_only)
            await asyncio.sleep(2)  # Delay để tránh rate limit

    async def run(self, channels=None, today_only=CRAWL_TODAY_ONLY):
        """
        Chạy crawler
        
        Args:
            channels: danh sách channels (nếu None, sử dụng từ config)
        """
        await self.client.start()
        self.last_run_today_only = today_only
        
        try:
            # Sử dụng channels từ parameter hoặc từ config
            channels_to_crawl = channels or CHANNELS_TO_CRAWL
            
            if not channels_to_crawl:
                logger.warning("⚠ Không có channels để crawl. Nhập danh sách channels:")
                logger.warning("  python telegram_crawler.py channel1 channel2 channel3")
                return
            
            # Crawl từ các channels
            await self.crawl_multiple_channels(
                channels_to_crawl,
                limit=CRAWL_LIMIT,
                today_only=today_only,
            )
            
        finally:
            await self.client.disconnect()

    def close(self):
        """Đóng các kết nối đồng bộ còn lại."""
        if self.mongo_client is not None:
            self.mongo_client.close()
            self.mongo_client = None
            self.db = None
            self.collection = None

    def display_stats(self):
        """Hiển thị thống kê"""
        logger.info("\n" + "="*50)
        logger.info("📊 THỐNG KÊ CRAWL")
        logger.info("="*50)

        if self.session_stats:
            total_new_messages = sum(item["new_messages"] for item in self.session_stats.values())
            total_skipped_messages = sum(item["skipped_messages"] for item in self.session_stats.values())
            logger.info(f"Số sources đã chạy: {len(self.session_stats)}")
            logger.info(f"Tổng tin nhắn mới trong phiên: {total_new_messages}")
            logger.info(f"Tổng tin nhắn bị bỏ qua trong phiên: {total_skipped_messages}")

            for channel_id, stats in self.session_stats.items():
                logger.info(
                    "  Channel %s (%s): mới=%s | skip=%s",
                    channel_id,
                    stats["channel_username"],
                    stats["new_messages"],
                    stats["skipped_messages"],
                )
                if stats.get("output_path"):
                    logger.info(f"    File local: {stats['output_path']}")

        if self.collection is not None:
            try:
                total_messages = self.collection.count_documents({})
                channels = self.collection.distinct("channel_id")
                logger.info(f"Tổng tin nhắn đang có trong DB: {total_messages}")
                logger.info(f"Số channels đang có dữ liệu trong DB: {len(channels)}")
                return
            except Exception as e:
                logger.warning(f"⚠ Không thể đọc thống kê từ MongoDB: {str(e)}")
        else:
            logger.info("MongoDB: không kết nối, chỉ hiển thị thống kê của phiên chạy hiện tại")

        if self.session_stats:
            mode = "hôm nay" if self.last_run_today_only else "mọi ngày"
            logger.info(f"Chế độ crawl: {mode} | Timezone: {CRAWL_TIMEZONE}")


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Telegram crawler")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--today", action="store_true", help="Chi crawl bai cua hom nay")
    mode_group.add_argument("--all", action="store_true", help="Crawl tat ca bai")
    parser.add_argument("channels", nargs="*", help="Danh sach channel can crawl")
    return parser.parse_args()


async def main():
    args = parse_cli_args()
    channels = args.channels
    today_only = CRAWL_TODAY_ONLY
    if args.today:
        today_only = True
    elif args.all:
        today_only = False
    
    crawler = TelegramCrawler()
    
    try:
        # Chạy crawler
        await crawler.run(channels=channels, today_only=today_only)
        
        # Hiển thị thống kê
        crawler.display_stats()
    finally:
        crawler.close()


if __name__ == "__main__":
    asyncio.run(main())
