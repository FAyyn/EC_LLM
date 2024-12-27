from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import MinHashLSH
from pyspark.sql.functions import col, explode, split

# 初始化SparkSession
spark = SparkSession.builder.appName("MarkdownDeduplication").getOrCreate()

# 读取Markdown文件
markdown_dir = "path/to/your/markdown/files"  # 替换为你的Markdown文件目录
df = spark.read.text(markdown_dir).toDF("text")

# 将Markdown文本转换为纯文本
df = df.withColumn("text", split(col("text"), "\n").getItem(0))

# 使用Tokenizer将文本数据转化为词袋表示
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(df)

# 使用HashingTF将词袋表示转化为特征向量
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# 使用MinHashLSH计算特征向量的哈希值
mh = MinHashLSH(inputCol="rawFeatures", outputCol="hashes", numHashTables=10)
model = mh.fit(featurizedData)
transformed = model.transform(featurizedData)

# 通过比较哈希值进行去重
deduplicated = transformed.dropDuplicates(["hashes"])

# 保存去重后的结果
output_path = "/workspace/MinerU-DATA/1/auto"  # 替换为你的输出目录
deduplicated.select("text").write.text(output_path)

# 停止SparkSession
spark.stop()