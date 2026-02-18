import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, to_timestamp, date_format,
    conv, abs as spark_abs, hash as spark_hash
)
from pyspark.sql.types import (
    StructType, StructField,
    LongType, StringType, TimestampType
)


BASE_DIR   = "/home/anonymous/code/Mining_Massive_Dataset"
INPUT_PATH = os.path.join(BASE_DIR, "tweets_clean.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "tweets_final.parquet")



FINAL_SCHEMA = StructType([
    StructField("tweet_id",    LongType(),      nullable=False),  # ID tweet
    StructField("user_id",     LongType(),      nullable=False),  # ID người dùng
    StructField("text",        StringType(),    nullable=False),  # Nội dung tweet
    StructField("timestamp",   TimestampType(), nullable=False),  # Thời gian đăng
    StructField("embeddings",  StringType(),    nullable=True),   # Placeholder cho NLP
    StructField("topic_label", StringType(),    nullable=True),   # Placeholder cho ML
    StructField("date",        StringType(),    nullable=False),  # Partition column
])


def create_spark_session():
    """Tạo Spark session với cấu hình phù hợp cho dữ liệu ~40GB."""
    spark = (
        SparkSession.builder
        .appName("SchemaDesign_DatePartition")
        # Memory config
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        # Shuffle & adaptive query
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Parquet output
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def build_final_schema(df):

    print("\n[1] Parsing timestamp từ tweetcreatedts ...")
    df = df.withColumn(
        "timestamp",
        to_timestamp(col("tweetcreatedts"), "yyyy-MM-dd HH:mm:ss")
    )

    # Fallback sang format ISO nếu parse lần đầu thất bại
    from pyspark.sql.functions import when
    df = df.withColumn(
        "timestamp",
        when(
            col("timestamp").isNull(),
            to_timestamp(col("tweetcreatedts"), "yyyy-MM-dd'T'HH:mm:ss")
        ).otherwise(col("timestamp"))
    )

    # Lọc bỏ dòng có timestamp null (không parse được)
    before = df.count()
    df = df.filter(col("timestamp").isNotNull())
    after = df.count()
    print(f"   Dropped {before - after:,} rows with unparseable timestamps")
    print(f"   Remaining: {after:,} rows")

    print("\n[2] Tạo cột date (YYYY-MM-DD) từ timestamp ...")
    df = df.withColumn("date", date_format(col("timestamp"), "yyyy-MM-dd"))

    print("\n[3] Tạo user_id từ hash(username) ...")
  
    df = df.withColumn(
        "user_id",
        spark_abs(spark_hash(col("username"))).cast(LongType())
    )

    print("\n[4] Select & cast sang final schema ...")
    df_final = df.select(
        col("tweetid").cast(LongType()).alias("tweet_id"),
        col("user_id"),
        col("text"),
        col("timestamp"),
        lit(None).cast(StringType()).alias("embeddings"),   # placeholder
        lit(None).cast(StringType()).alias("topic_label"),  # placeholder
        col("date"),
    )

    # Lọc tweet_id null sau khi cast (dữ liệu xấu)
    df_final = df_final.filter(col("tweet_id").isNotNull())

    return df_final


def describe_dataset(df, label="Dataset"):
    """In thống kê tóm tắt về dataset."""
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    count = df.count()
    print(f"  Tổng số tweets : {count:,}")

    # Phân bố theo ngày
    print(f"\n  Phân bố theo ngày (top 10):")
    df.groupBy("date").count() \
      .orderBy("date") \
      .show(10, truncate=False)

    # Date range
    from pyspark.sql.functions import min as spark_min, max as spark_max
    stats = df.agg(
        spark_min("date").alias("start_date"),
        spark_max("date").alias("end_date")
    ).collect()[0]
    print(f"  Date range: {stats['start_date']}  →  {stats['end_date']}")
    print()


def save_partitioned(df, output_path):
    """Lưu DataFrame dạng Parquet, partition theo cột date."""
    print(f"\n[5] Ghi Parquet → {output_path}  (partitionBy date) ...")

    df.write \
        .mode("overwrite") \
        .partitionBy("date") \
        .parquet(output_path)

    print("   ✓ Ghi xong!")

    # Kiểm tra số lượng partition
    partition_dirs = [
        d for d in os.listdir(output_path)
        if d.startswith("date=")
    ]
    print(f"   Số partition (ngày): {len(partition_dirs)}")
    print(f"   Ví dụ partition: {sorted(partition_dirs)[:5]}")


def verify_output(spark, output_path):
    """Đọc lại output và in schema + stats để xác nhận."""
    print(f"\n{'═'*60}")
    print("  VERIFICATION — Đọc lại output Parquet")
    print(f"{'═'*60}")

    df_check = spark.read.parquet(output_path)

    print("\n  Schema:")
    df_check.printSchema()

    print(f"  Tổng rows: {df_check.count():,}")
    print("\n  Sample data (5 dòng đầu):")
    df_check.select(
        "tweet_id", "user_id", "text", "timestamp", "date"
    ).show(5, truncate=60)

    print("\n  Null check:")
    from pyspark.sql.functions import count, when
    df_check.select([
        count(when(col(c).isNull(), c)).alias(c)
        for c in ["tweet_id", "user_id", "text", "timestamp", "date"]
    ]).show()

    print("\n  Tweets per day (all dates):")
    df_check.groupBy("date").count().orderBy("date").show(100, truncate=False)


def main():

    print(f"\nInput  : {INPUT_PATH}")
    print(f"Output : {OUTPUT_PATH}\n")

    spark = create_spark_session()

    print("Loading tweets_clean.parquet ...")
    df_raw = spark.read.parquet(INPUT_PATH)
    print(f"Đọc được {df_raw.count():,} rows")
    print("Columns hiện tại:", df_raw.columns)

    df_final = build_final_schema(df_raw)

    describe_dataset(df_final, label="Final Dataset — trước khi ghi")

    save_partitioned(df_final, OUTPUT_PATH)

    verify_output(spark, OUTPUT_PATH)

    print("\n" + "═"*60)
    print("  ✓ PIPELINE HOÀN THÀNH!")
    print(f"  Output: {OUTPUT_PATH}")
    print("  Schema: tweet_id | user_id | text | timestamp | embeddings | topic_label")
    print("  Partition: date (YYYY-MM-DD)")
    print("═"*60 + "\n")

    spark.stop()


if __name__ == "__main__":
    main()
