"""
Utility script để sử dụng TelegramCrawler
"""
import asyncio
import argparse
from telegram_crawler import TelegramCrawler
from pymongo import MongoClient
import json
from config import (
    MONGO_URI,
    MONGO_DB_NAME,
    MONGO_COLLECTION_NAME,
    CHANNELS_TO_CRAWL,
    CRAWL_LIMIT,
    CRAWL_TODAY_ONLY,
)


def query_messages(channel_id=None, limit=10):
    """
    Truy vấn tin nhắn từ MongoDB
    
    Args:
        channel_id: ID của channel (None = tất cả)
        limit: số lượng kết quả
    """
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    
    query = {} if channel_id is None else {"channel_id": channel_id}
    messages = list(collection.find(query).sort("_id", -1).limit(limit))
    
    print(f"\n📨 Tìm thấy {len(messages)} tin nhắn:\n")
    
    for msg in messages:
        print(f"ID: {msg['message_id']} | Channel: {msg['channel_id']}")
        print(f"Text: {msg['text'][:100]}..." if len(msg['text']) > 100 else f"Text: {msg['text']}")
        print(f"Time: {msg['timestamp']}")
        
        if msg.get('forward_from'):
            print(f"Forward from: {msg['forward_from']}")
        
        if msg.get('media'):
            print(f"Media: {msg['media']['type']} - {msg['media']['file_name']}")
        
        print("-" * 80)
    
    client.close()


def export_to_json(filename="messages_export.json"):
    """Xuất dữ liệu ra JSON"""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    
    messages = list(collection.find({}))
    
    # Convert ObjectId to string
    for msg in messages:
        msg['_id'] = str(msg['_id'])
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Đã xuất {len(messages)} tin nhắn ra {filename}")
    client.close()


async def crawl_from_list(channels_list, today_only=CRAWL_TODAY_ONLY):
    """
    Crawl từ danh sách channels
    
    Args:
        channels_list: danh sách tên channels (['channel1', 'channel2', ...])
    """
    crawler = TelegramCrawler()
    
    await crawler.client.start()
    
    try:
        for channel in channels_list:
            print(f"\n{'='*50}")
            print(f"🔍 Crawling: {channel}")
            print(f"{'='*50}")
            await crawler.crawl_channel(channel, limit=CRAWL_LIMIT, today_only=today_only)
            await asyncio.sleep(3)
        
        # Hiển thị thống kê
        crawler.display_stats()
        
    finally:
        await crawler.client.disconnect()
        if crawler.mongo_client is not None:
            crawler.mongo_client.close()


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Telegram crawler utility")
    subparsers = parser.add_subparsers(dest="command")

    crawl_parser = subparsers.add_parser("crawl")
    crawl_mode_group = crawl_parser.add_mutually_exclusive_group()
    crawl_mode_group.add_argument("--today", action="store_true", help="Chi crawl bai cua hom nay")
    crawl_mode_group.add_argument("--all", action="store_true", help="Crawl tat ca bai")
    crawl_parser.add_argument("channels", nargs="*", help="Danh sach channel can crawl")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("channel_id", nargs="?", type=int)
    query_parser.add_argument("limit", nargs="?", type=int, default=10)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("filename", nargs="?", default="messages_export.json")

    return parser.parse_args()


# ============= HƯỚNG DẪN SỬ DỤNG =============
if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════╗
    ║   TELEGRAM CRAWLER - UTILITY SCRIPT       ║
    ╚════════════════════════════════════════════╝
    
    Các cách sử dụng:
    
    1. Crawl từ danh sách channels:
       python utils.py crawl channel1 channel2 channel3
       
    2. Truy vấn tin nhắn từ DB:
       python utils.py query [channel_id] [limit]
       
    3. Xuất dữ liệu ra JSON:
       python utils.py export [filename]
    
    Ví dụ:
       python utils.py crawl news hongkong tech
       python utils.py query 123456789 20
       python utils.py export data.json
    """)

    args = parse_cli_args()

    if args.command == "crawl":
        channels = args.channels if args.channels else CHANNELS_TO_CRAWL
        today_only = CRAWL_TODAY_ONLY
        if args.today:
            today_only = True
        elif args.all:
            today_only = False

        if channels:
            asyncio.run(crawl_from_list(channels, today_only=today_only))
        else:
            print("❌ Vui lòng nhập ít nhất một channel")

    elif args.command == "query":
        query_messages(args.channel_id, args.limit)

    elif args.command == "export":
        export_to_json(args.filename)

    else:
        print("❌ Command không hợp lệ")
