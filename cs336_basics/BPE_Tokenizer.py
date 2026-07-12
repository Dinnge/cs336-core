from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from tests.common import gpt2_bytes_to_unicode 
    
#     return bytes_count
def count_bytes(token_freq):
    pair_freq = Counter()
    for token, freq in token_freq.items():
        # 跳过长度为 1 的 token，会快个0.2s
        if len(token) < 2:
            continue
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            
            pair_freq[pair] += freq
    return pair_freq

def merge_bytes( #要返回新的计数和合并之后的token id对照表
    freq: Counter,
    new_ids: int, 
    tokens_freq,
    merge: dict[int, bytes],
):
    if not freq:
        return None, freq, merge, tokens_freq
    best_pair = max(freq.items(), key=lambda item: (item[1], item[0]))[0]
    # best_pair = freq.most_common(1)[0][0]

    a, b = best_pair
    del freq[best_pair]
    merge[new_ids] = a + b

    for token, freq_count in list(tokens_freq.items()):
        merged = []
        i = 0
        changed = False
        while i < len(token):
            if i + 1 < len(token) and token[i] == a and token[i+1] == b:
                merged.append(a + b)
                i += 2
                changed = True
            else:
                merged.append(token[i])
                i += 1
        
        if changed:
            merged_token = tuple(merged)
            if len(token) >= 2:
                for j in range(len(token) - 1):
                    old_pair = (token[j], token[j+1])
                    freq[old_pair] -= freq_count
                    if freq[old_pair] <= 0:
                        freq.pop(old_pair, None)
            
            if len(merged_token) >= 2:
                for j in range(len(merged_token) - 1):
                    new_pair = (merged_token[j], merged_token[j+1])
                    freq[new_pair] += freq_count
            
            del tokens_freq[token]
            tokens_freq[merged_token] += freq_count
    
    return best_pair, freq, merge, tokens_freq

def process_chunk(args):
    file_path, start, end, special_token = args
    with open(file_path, "rb") as f:   # 自己打开文件
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
    tokens_freq = defaultdict(int)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    parts = chunk.split(special_token)
    for i, part in enumerate(parts):
        if i > 0:
            # 遇到特殊 token 了，把它作为一个完整 token 加进去
            # 但不参与合并统计
            tokens_freq[(special_token.encode("utf-8"),)] += 1
        
        for token in re.findall(PAT, part):
            token_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
            tokens_freq[token_tuple] += 1

    return tokens_freq

# 单进程的
def BPE_Tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    special_tokens_bytes = [tokens.encode('utf-8') for tokens in special_tokens]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes[0])
        byte_to_unicode = gpt2_bytes_to_unicode()
        merge = {}
        merge[0] = special_tokens_bytes[0]
        for i in range(256):
            merge[i+1] = bytes([list(byte_to_unicode.keys())[i]])
        # 把所有的token记录下来天浪费时间了
        # all_tokens = []
        token_freq = defaultdict(int)
        
        # The following is a parallel implementation using multiprocessing.
        chunk_args = [
            (input_path, start, end, special_tokens[0])
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        num_workers = min(cpu_count(), len(chunk_args))
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_chunk, chunk_args)
        
        # 合并所有进程的结果
        for chunk_tokens_freq in results:
            for token_tuple, freq in chunk_tokens_freq.items():
                token_freq[token_tuple] += freq
        
        pairs = count_bytes(token_freq)
        # Run pre-tokenization on your chunk and store the counts for each pre-token

        best_pairs = []
        while len(merge) < vocab_size and pairs:
            pair, pairs, merge, token_freq = merge_bytes(pairs, len(merge), token_freq, merge)
            if pair is None:
                break
            # if len(merge) == vocab_size - 1:
            #     print(pair)
            best_pairs.append(pair)
 
    return merge, best_pairs

if __name__ == "__main__":
    import json
    import os
    import time

    byte_to_unicode = gpt2_bytes_to_unicode()
    def bytes_to_gpt2_str(b: bytes) -> str:
        """将 bytes 转换为 GPT-2 格式的字符串表示"""
        return "".join(byte_to_unicode[byte] for byte in b)
    start = time.time()
    vocab, merges = BPE_Tokenizer(
        "tests/fixtures/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"]
    )
    elapsed = time.time() - start
    print(f"Training time: {elapsed:.2f}s")
    print(f"vocab size: {len(vocab)}, merges: {len(merges)}")

    # 找最长 token
    longest_id = max(vocab, key=lambda k: len(vocab[k]))
    longest_bytes = vocab[longest_id]
    print(f"Longest token (id={longest_id}, len={len(longest_bytes)}): {longest_bytes}")

    merges_txt_lines = []
    for pair in merges:
        a_str = bytes_to_gpt2_str(pair[0])
        b_str = bytes_to_gpt2_str(pair[1])
        merges_txt_lines.append(f"{a_str} {b_str}")
    merges_txt = "\n".join(merges_txt_lines)
    with open("merges_tinystory.txt", "w", encoding="utf-8") as f:
        f.write(merges_txt)

    # vocab 是 {int: bytes}，反转成 GPT-2 格式 {str: int}
    vocab_json = {}
    for token_id, token_bytes in vocab.items():
        vocab_json[bytes_to_gpt2_str(token_bytes)] = token_id

    output_path = os.path.join(os.path.dirname(__file__), "vocab_tinystory.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=4)