# Telegram Topic Classification Summary

## Source

- Database: `telegram_data`
- Collection: `messages`
- Classification field: `war_classification.label`
- Classifier source: Grok/xAI classification script from `F:\Crawl_News_Iran\mongo\classify_telegram_messages.py`

## Label counts

| Label | Count | Share |
| --- | ---: | ---: |
| `unrelated` | 5,501 | 46.35% |
| `russia_ukraine_war` | 3,482 | 29.34% |
| `us_iran_war` | 2,885 | 24.31% |

Total classified Telegram messages: `11,868`.

## Interpretation

- `russia_ukraine_war`: Telegram messages about Russia-Ukraine conflict, military operations, strikes, frontlines, sanctions, or related escalation.
- `us_iran_war`: Telegram messages about US-Iran conflict, Iran-related military escalation, strikes, threats, or diplomatic crisis involving explicit US role.
- `unrelated`: Messages that do not clearly belong to either conflict topic.

## How it connects to the LSH pipeline

The Telegram classifier acts as an upstream filtering step before similarity search.

Recommended flow:

`Telegram crawl -> Grok topic classification -> export canonical Parquet with topic_label -> filter topic -> shingling -> MinHash/LSH -> narrative clusters`

This makes the LSH stage cleaner because it can run separately on:

- Russia-Ukraine Telegram messages
- US-Iran Telegram messages
- combined Twitter + Telegram samples

## Local configuration

Keep Mongo credentials in `.env`, not in code:

```env
MONGO_URI=...
MONGO_DB_NAME=telegram_data
MONGO_COLLECTION_NAME=messages
```

Then regenerate this summary with:

```bash
python scripts/reporting/summarize_telegram_topics.py
```
