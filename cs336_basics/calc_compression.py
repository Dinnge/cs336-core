"""计算 TinyStories tokenizer 的压缩比"""
import sys
import random
import time

sys.path.insert(0, r"C:\Users\Dy.ming\Desktop\CS336\assignment1-basics")
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path


def sample_documents(filepath: str, num_docs: int = 10) -> list[str]:
    """
    从数据集中随机采样 num_docs 篇文档。
    文档以 <|endoftext|> 分隔。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        # 用 <|endoftext|> 分割，去掉空文档
        docs = [d.strip() for d in f.read().split("<|endoftext|>")]
        docs = [d for d in docs if d]  # 去掉空字符串

    print(f"Total documents in dataset: {len(docs)}")
    sampled = random.sample(docs, min(num_docs, len(docs)))
    return sampled


def calc_compression_ratio(docs: list[str], tokenizer) -> tuple[float, list[float]]:
    """
    计算压缩比 = 原始字节数 / token 数量。
    返回 (平均压缩比, 每个文档的压缩比列表)
    """
    ratios = []
    for i, doc in enumerate(docs):
        num_bytes = len(doc.encode("utf-8"))
        tokens = tokenizer.encode(doc)
        num_tokens = len(tokens)
        ratio = num_bytes / num_tokens
        ratios.append(ratio)
        print(f"  Doc {i+1}: {num_bytes} bytes -> {num_tokens} tokens, ratio = {ratio:.2f}")

    avg_ratio = sum(ratios) / len(ratios)
    return avg_ratio, ratios


if __name__ == "__main__":
    random.seed(42)

    # 加载 tokenizer
    vocab_path = "cs336_basics/vocab_tinystory.json"
    merges_path = "merges_tinystory.txt"
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=["<|endoftext|>"],
    )

    # 采样文档
    train_path = "tests/fixtures/TinyStoriesV2-GPT4-train.txt"
    print(f"Sampling 10 documents from {train_path}...")
    docs = sample_documents(train_path, num_docs=10)

    # 计算压缩比
    print("\nCompression ratio per document:")
    avg_ratio, ratios = calc_compression_ratio(docs, tokenizer)
    print(f"\nAverage compression ratio (bytes/token): {avg_ratio:.2f}")

    # 顺便测吞吐量
    print("\n--- Throughput test ---")
    # 取前 1MB 文本测速
    with open(train_path, "r", encoding="utf-8") as f:
        sample_text = f.read(1_000_000)  # ~1MB
    num_bytes = len(sample_text.encode("utf-8"))
    start = time.time()
    tokens = tokenizer.encode(sample_text)
    elapsed = time.time() - start
    throughput = num_bytes / elapsed
    print(f"Encoded {num_bytes:,} bytes in {elapsed:.2f}s")
    print(f"Throughput: {throughput:,.0f} bytes/s ({throughput/1e6:.2f} MB/s)")

    # Pile 数据集 825GB 需要多久
    pile_size = 825 * 1024 * 1024 * 1024  # bytes
    pile_seconds = pile_size / throughput
    pile_hours = pile_seconds / 3600
    print(f"Estimated time to tokenize Pile (825GB): {pile_hours:.1f} hours")
