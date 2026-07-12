input = [b'e', b' ', b'c', b'a', b't']
tmpinput = b"".join(input)
print(tmpinput)

# @ext:Tencent-Cloud.coding-copilot .enableFloatShortcut

from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter

def count_bytes(
    input: list[list[bytes]]
):
    bytes_count = Counter()
    for token in input:
        i = 0
        while i < len(token) - 1:
            cur = token[i]
            nxt = token[i + 1]
            bytes_count[(cur, nxt)] += 1
            i += 1
    
    return bytes_count

def merge_bytes( #要返回新的计数和合并之后的token id对照表
    freq: Counter,
    new_ids: int,
    tokens: list[list[bytes]],
    merge: dict[bytes, int],
):  
    if not freq:
        return None, freq, merge, tokens
    best_pair = max(freq.items(), key=lambda item: (item[1], item[0]))[0]

    a, b = best_pair
    del freq[best_pair]
    # del merge[best_pair] 这个不能够删除
    merge[a + b] = new_ids

    merged_tokens = []
    for token in tokens:
        # 处理单个 token
        merged_token = []
        i = 0
        while i < len(token):
            # 检查是否可以合并当前位置和下一个位置的字节
            if i + 1 < len(token) and token[i] == a and token[i + 1] == b:
                merged_token.append(a + b)  # 合并成新 token
                i += 2
            else:
                merged_token.append(token[i])
                i += 1
        merged_tokens.append(merged_token)

    new_freq = count_bytes(merged_tokens)

    return best_pair, new_freq, merge, merged_tokens

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

        merge = {bytes([i]): i for i in range(256)}
        all_tokens = []

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # 要转成单字节
            tokens = re.findall(PAT, chunk)
            for token in tokens:
                # 把单个 token 的字节收集成一个列表
                token_bytes = [bytes([b]) for b in token.encode("utf-8") if b != 32]
                # 把这个 token 的字节列表作为一个整体 append
                all_tokens.append(token_bytes)
        
        pairs = count_bytes(all_tokens)
        # Run pre-tokenization on your chunk and store the counts for each pre-token

        vocab_size -= len(special_tokens)

        best_pairs = []
        while len(merge) < vocab_size and pairs:
            pair, pairs, merge, all_tokens = merge_bytes(pairs, len(merge), all_tokens, merge)
            if pair is None:
                break
            if len(merge) == vocab_size - 1:
                print(pair)
            best_pairs.append(pair)

        merge[special_tokens_bytes[0]] = vocab_size
    
    return merge, best_pairs

if __name__ == "__main__":
    vocab, merges = BPE_Tokenizer(
        "tests/fixtures/corpus.en",
        500,
        ["<|endoftext|>"]
    )
    print(f"vocab size: {len(vocab)}")
    print(f"merges: {len(merges)}")
# # 并行训练
# def BPE_Tokenizer(
#     input_path: str,
#     vocab_size: int,
#     special_tokens: list[str],
# ):
#     special_tokens_bytes = [tokens.encode('utf-8') for tokens in special_tokens]

#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#     with open(input_path, "rb") as f:
#         num_processes = 8
#         boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes)

#         merge = {bytes([i]): i for i in range(256)}
#         best_pairs = []

#         # The following is a serial implementation, but you can parallelize this
#         # by sending each start/end pair to a set of processes.
#         for start, end in zip(boundaries[:-1], boundaries[1:]):
#             f.seek(start)
#             chunk = f.read(end - start).decode("utf-8", errors="ignore")
#             chunk_merge = {}

#             # 要转成单字节
#             tokens = re.findall(PAT, chunk)
#             tokens_bytes = [bytes([b]) for token in tokens for b in token.encode("utf-8")]

#             pairs = count_bytes(tokens_bytes)
#             # Run pre-tokenization on your chunk and store the counts for each pre-token

#             while len(chunk_merge) + len(merge) < vocab_size:
#                 pair, pairs, chunk_merge, tokens_bytes = merge_bytes(pairs, len(chunk_merge) + len(merge), tokens_bytes, merge)
#                 best_pairs.append(pair)
            
#             merge.update(chunk_merge)
    
#     return merge, best_pairs

    # vocab_reversed = {id_: token for token, id_ in vocab.items()}
    
    # return vocab_reversed, merges
    # raise NotImplementedError





from regex import A
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter, defaultdict
from tests.common import gpt2_bytes_to_unicode 

def count_bytes(
    input: list[list[bytes]]
):
    bytes_count = Counter()
    for token in input:
        i = 0
        while i < len(token) - 1:
            cur = token[i]
            nxt = token[i + 1]
            bytes_count[(cur, nxt)] += 1
            i += 1
    
    return bytes_count

def merge_bytes( #要返回新的计数和合并之后的token id对照表
    freq: Counter,
    new_ids: int, 
    tokens: list[list[bytes]],
    merge: dict[bytes, int],
):
    if not freq:
        return None, freq, merge, tokens
    best_pair = max(freq.items(), key=lambda item: (item[1], item[0]))[0]

    a, b = best_pair
    del freq[best_pair]
    merge[new_ids] = a + b

    # 这种方法太慢了
    merged_tokens = []
    for token in tokens:
        merged_token = []
        i = 0
        while i < len(token):
            if i + 1 < len(token) and token[i] == a and token[i + 1] == b:
                merged_token.append(a + b)
                # if i + 2 < len(token) and (i + 3 >= len(token) or token[i + 2] != a or token[i + 3] != b):
                #     freq[a + b, token[i+2]] += 1
                #     del freq[b, token[i+2]]
                # if i:
                #     del freq[token[i-1], a]
                i += 2
            else:
                merged_token.append(token[i])
                i += 1
        merged_tokens.append(merged_token)

    new_freq = count_bytes(merged_tokens)

    return best_pair, new_freq, merge, merged_tokens

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
        all_tokens = []
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # 要转成单字节
            parts = chunk.split(special_tokens[0])
    
            for i, part in enumerate(parts):
                if i > 0:
                    # 遇到特殊 token 了，把它作为一个完整 token 加进去
                    # 但不参与合并统计
                    all_tokens.append([special_tokens_bytes[0]])
                
                # 对普通文本部分做预分词
                tokens = re.findall(PAT, part)
                for token in tokens:
                    token_bytes = [bytes([b]) for b in token.encode("utf-8")]
                    all_tokens.append(token_bytes)
        
        pairs = count_bytes(all_tokens)
        # Run pre-tokenization on your chunk and store the counts for each pre-token

        best_pairs = []
        while len(merge) < vocab_size and pairs:
            pair, pairs, merge, all_tokens = merge_bytes(pairs, len(merge), all_tokens, merge)
            if pair is None:
                break
            best_pairs.append(pair)

    return merge, best_pairs

if __name__ == "__main__":
    vocab, merges = BPE_Tokenizer(
        "tests/fixtures/tinystories_sample_5M.txt",
        1000,
        ["<|endoftext|>"]
    )
    print(f"vocab size: {len(vocab)}")
    print(f"merges: {len(merges)}")







# # 并行训练
# def BPE_Tokenizer(
#     input_path: str,
#     vocab_size: int,
#     special_tokens: list[str],
# ):
#     special_tokens_bytes = [tokens.encode('utf-8') for tokens in special_tokens]

#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#     with open(input_path, "rb") as f:
#         num_processes = 8
#         boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes)

#         merge = {bytes([i]): i for i in range(256)}
#         best_pairs = []

#         # The following is a serial implementation, but you can parallelize this
#         # by sending each start/end pair to a set of processes.
#         for start, end in zip(boundaries[:-1], boundaries[1:]):
#             f.seek(start)
#             chunk = f.read(end - start).decode("utf-8", errors="ignore")
#             chunk_merge = {}

#             # 要转成单字节
#             tokens = re.findall(PAT, chunk)
#             tokens_bytes = [bytes([b]) for token in tokens for b in token.encode("utf-8")]

#             pairs = count_bytes(tokens_bytes)
#             # Run pre-tokenization on your chunk and store the counts for each pre-token

#             while len(chunk_merge) + len(merge) < vocab_size:
#                 pair, pairs, chunk_merge, tokens_bytes = merge_bytes(pairs, len(chunk_merge) + len(merge), tokens_bytes, merge)
#                 best_pairs.append(pair)
            
#             merge.update(chunk_merge)
    
#     return merge, best_pairs
# import regex as re

# def train_bpe(
#     input_path: str,
#     vocab_size: int,
#     special_tokens: list[str],
# ):
#     special_tokens_bytes = [token.encode('utf-8') for token in special_tokens]
    
#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
#     with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
#         text = f.read()
    
#     text_chunks = re.findall(PAT, text)
    
#     id_to_bytes = []
#     bytes_to_id = {}
    
#     for token_bytes in special_tokens_bytes:
#         bytes_to_id[token_bytes] = len(id_to_bytes)
#         id_to_bytes.append(token_bytes)
    
#     for b in range(256):
#         byte_token = bytes([b])
#         bytes_to_id[byte_token] = len(id_to_bytes)
#         id_to_bytes.append(byte_token)
    
#     if len(id_to_bytes) >= vocab_size:
#         vocab = {i: id_to_bytes[i] for i in range(vocab_size)}
#         return vocab, []
    
#     token_freq = {}
#     for chunk in text_chunks:
#         chunk_bytes = chunk.encode('utf-8')
#         token_ids = tuple(bytes_to_id[bytes([b])] for b in chunk_bytes)
#         if token_ids in token_freq:
#             token_freq[token_ids] += 1
#         else:
#             token_freq[token_ids] = 1
    
#     pairs = {}
#     for token, freq in token_freq.items():
#         for i in range(len(token) - 1):
#             pair = (token[i], token[i + 1])
#             if pair in pairs:
#                 pairs[pair] += freq
#             else:
#                 pairs[pair] = freq
    
#     merges = []
    
#     num_merges = vocab_size - len(id_to_bytes)
#     pair_key_fn = lambda item: (item[1], id_to_bytes[item[0][0]], id_to_bytes[item[0][1]])
    
#     for _ in range(num_merges):
#         if not pairs:
#             break
        
#         best_pair = max(pairs.items(), key=pair_key_fn)[0]
#         new_id = len(id_to_bytes)
        
#         merged_bytes = id_to_bytes[best_pair[0]] + id_to_bytes[best_pair[1]]
#         bytes_to_id[merged_bytes] = new_id
#         id_to_bytes.append(merged_bytes)
#         merges.append((id_to_bytes[best_pair[0]], id_to_bytes[best_pair[1]]))
        
#         a, b = best_pair
#         new_token_freq = {}
        
#         for token, freq in token_freq.items():
#             has_a = False
#             for x in token:
#                 if x == a:
#                     has_a = True
#                     break
            
#             if not has_a:
#                 if token in new_token_freq:
#                     new_token_freq[token] += freq
#                 else:
#                     new_token_freq[token] = freq
#                 continue
            
#             new_token = []
#             i = 0
#             n = len(token)
#             while i < n:
#                 if i + 1 < n and token[i] == a and token[i + 1] == b:
#                     new_token.append(new_id)
#                     i += 2
#                 else:
#                     new_token.append(token[i])
#                     i += 1
#             new_token_tuple = tuple(new_token)
#             if new_token_tuple in new_token_freq:
#                 new_token_freq[new_token_tuple] += freq
#             else:
#                 new_token_freq[new_token_tuple] = freq
        
#         token_freq = new_token_freq
        
#         pairs = {}
#         for token, freq in token_freq.items():
#             t0 = token[0]
#             for i in range(1, len(token)):
#                 t1 = token[i]
#                 pair = (t0, t1)
#                 if pair in pairs:
#                     pairs[pair] += freq
#                 else:
#                     pairs[pair] = freq
#                 t0 = t1
    
#     vocab = {i: id_to_bytes[i] for i in range(len(id_to_bytes))}
#     return vocab, merges

# if __name__ == "__main__":
#     vocab, merges = train_bpe(
#         "tests/fixtures/corpus.en",
#         500,
#         ["<|endoftext|>"]
#     )
#     print(f"vocab size: {len(vocab)}")
#     print(f"merges count: {len(merges)}")
#     print(f"first 5 merges: {merges[:5]}")



# def merge_bytes( #要返回新的计数和合并之后的token id对照表
#     freq: Counter,
#     new_ids: int, 
#     tokens: list[list[bytes]],
#     merge: dict[bytes, int],
# ):
#     if not freq:
#         return None, freq, merge, tokens
#     best_pair = max(freq.items(), key=lambda item: (item[1], item[0]))[0]

#     a, b = best_pair
#     del freq[best_pair]
#     merge[new_ids] = a + b

#     # 这种方法太慢了
#     merged_tokens = []
#     for token in tokens:
#         merged_token = []
#         i = 0
#         while i < len(token):
#             if i + 1 < len(token) and token[i] == a and token[i + 1] == b:
#                 merged_token.append(a + b)
#                 # if i + 2 < len(token) and (i + 3 >= len(token) or token[i + 2] != a or token[i + 3] != b):
#                 #     freq[a + b, token[i+2]] += 1
#                 #     del freq[b, token[i+2]]
#                 # if i:
#                 #     del freq[token[i-1], a]
#                 i += 2
#             else:
#                 merged_token.append(token[i])
#                 i += 1
#         merged_tokens.append(merged_token)

#     new_freq = count_bytes(merged_tokens)

#     return best_pair, new_freq, merge, merged_tokens



# 现在的freq是字节tuple和对应出现的次数
# def merge_bytes( #要返回新的计数和合并之后的token id对照表
#     freq: Counter,
#     new_ids: int, 
#     tokens_freq,
#     merge: dict[int, bytes],
# ):
#     if not freq:
#         return None, freq, merge, tokens_freq
#     best_pair = max(freq.items(), key=lambda item: (item[1], item[0]))[0]
#     # best_pair = freq.most_common(1)[0][0]

#     a, b = best_pair
#     del freq[best_pair]
#     merge[new_ids] = a + b

#     new_tokens_freq = defaultdict(int)
    
#     # 循环变量名改为 freq_count
#     for token, freq_count in tokens_freq.items():
#         merged = []
#         i = 0
#         while i < len(token):
#             if i + 1 < len(token) and token[i] == a and token[i+1] == b:
#                 merged.append(a + b)
#                 i += 2
#             else:
#                 merged.append(token[i])
#                 i += 1
        
#         #在 while 循环外面创建完整的 merged_token
#         merged_token = tuple(merged)
        
#         #使用 += 累加频率
#         new_tokens_freq[merged_token] = freq_count
    
#     new_freq = count_bytes(new_tokens_freq)
#     return best_pair, new_freq, merge, new_tokens_freq




# def count_bytes(
#     input: list[list[bytes]]
# ):
#     bytes_count = Counter()
#     for token in input:
#         i = 0
#         while i < len(token) - 1:
#             cur = token[i]
#             nxt = token[i + 1]
#             bytes_count[(cur, nxt)] += 1
#             i += 1


from regex import A
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter, defaultdict
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

def process_chunk(
    file_path: str,
    start, end, 
    special_tokens: list[str],
) -> tuple[int, int]:
    with open(file_path, "rb") as f:   # 自己打开文件
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
    tokens_freq = defaultdict(int)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    parts = chunk.split(special_tokens[0])
    for i, part in enumerate(parts):
        if i > 0:
            # 遇到特殊 token 了，把它作为一个完整 token 加进去
            # 但不参与合并统计
            tokens_freq[(special_tokens[0],)] += 1
        
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
        
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Windows 换行符 \r\n 统一转成 \n，避免 \r 参与 BPE 合并，要是用Linux就把这个去掉就好了
            chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
            # 要转成单字节
            parts = chunk.split(special_tokens[0])
    
            for i, part in enumerate(parts):
                if i > 0:
                    # 遇到特殊 token 了，把它作为一个完整 token 加进去
                    # 但不参与合并统计
                    token_freq[(special_tokens_bytes[0],)] += 1
                
                # 对普通文本部分做预分词
                tokens = re.findall(PAT, part)
                for token in tokens:
                    # 之前的版本，记录所有的token
                    # token_bytes = [bytes([b]) for b in token.encode("utf-8")]
                    # all_tokens.append(token_bytes)

                    # 转成 tuple（因为 list 不能当字典的 key）
                    token_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
                    # 字典的 key 是 token，value 是出现次数
                    token_freq[token_tuple] += 1
        
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

    start = time.time()
    vocab, merges = BPE_Tokenizer(
        "tests/fixtures/tinystories_sample_5M.txt",
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

    # vocab 是 {int: bytes}，反转成 {bytes字符串: int}
    vocab_json = {}
    for token_id, token_bytes in vocab.items():
        vocab_json[token_bytes.decode("utf-8", errors="replace")] = token_id

    output_path = os.path.join(os.path.dirname(__file__), "vocab.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=4)





from regex import A
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter, defaultdict
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
        
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Windows 换行符 \r\n 统一转成 \n，避免 \r 参与 BPE 合并，要是用Linux就把这个去掉就好了
            chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
            # 要转成单字节
            parts = chunk.split(special_tokens[0])
    
            for i, part in enumerate(parts):
                if i > 0:
                    # 遇到特殊 token 了，把它作为一个完整 token 加进去
                    # 但不参与合并统计
                    token_freq[(special_tokens_bytes[0],)] += 1
                
                # 对普通文本部分做预分词
                tokens = re.findall(PAT, part)
                for token in tokens:
                    # 之前的版本，记录所有的token
                    # token_bytes = [bytes([b]) for b in token.encode("utf-8")]
                    # all_tokens.append(token_bytes)

                    # 转成 tuple（因为 list 不能当字典的 key）
                    token_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))
                    # 字典的 key 是 token，value 是出现次数
                    token_freq[token_tuple] += 1
        
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

    vocab, merges = BPE_Tokenizer(
        "tests/fixtures/tinystories_sample_5M.txt",
        10000,
        ["<|endoftext|>"]
    )

    # vocab 是 {int: bytes}，反转成 {bytes字符串: int}
    vocab_json = {}
    for token_id, token_bytes in vocab.items():
        vocab_json[token_bytes.decode("utf-8", errors="replace")] = token_id

    output_path = os.path.join(os.path.dirname(__file__), "vocab.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=4)























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

    start = time.time()
    vocab, merges = BPE_Tokenizer(
        "tests/fixtures/tinystories_sample_5M.txt",
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

    # vocab 是 {int: bytes}，反转成 {bytes字符串: int}
    vocab_json = {}
    for token_id, token_bytes in vocab.items():
        vocab_json[token_bytes.decode("utf-8", errors="replace")] = token_id

    output_path = os.path.join(os.path.dirname(__file__), "vocab.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=4)





import json
import regex as re
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.merge_lookup = {}
        for pair in merges:
            a, b = pair
            self.merge_lookup.setdefault(a, []).append(pair)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        encoded_tokens = []
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        if self.special_tokens:
            sorted_tokens = sorted([t for t in self.special_tokens if t], key=len, reverse=True) # 这个地方一定要注意啊，正则匹配最长匹配原则
            pattern = "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
            if sorted_tokens:
                parts = re.split(pattern, text)
            else:
                parts = [text]
            for part in parts:
                if part == "":
                    continue
                if part in self.special_tokens:
                    encoded_tokens.append(self.vocab_reverse[part.encode("utf-8")])
                    continue
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                part = part.replace("\r\n", "\n").replace("\r", "\n")
                tokens = re.findall(PAT, part)
                for token in tokens:
                    raw_bytes = token.encode("utf-8")
                    token = [bytes([b]) for b in token.encode("utf-8")]  # 这个和tuple(token)不同
                    if raw_bytes in self.vocab_reverse:
                        encoded_tokens.append(self.vocab_reverse[raw_bytes])
                        continue
                    bytes_list = list(token)
                    for merge_pair in self.merges:
                        a, b = merge_pair
                        new_seq = []
                        i = 0
                        while i < len(bytes_list):
                            if i + 1 < len(bytes_list) and bytes_list[i] == a and bytes_list[i+1] == b:
                                new_seq.append(a + b)
                                i += 2
                            else:
                                new_seq.append(bytes_list[i])
                                i += 1
                        bytes_list = new_seq
                    for token_seq in bytes_list:
                        encoded_tokens.append(self.vocab_reverse[token_seq])
        else:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            part = text.replace("\r\n", "\n").replace("\r", "\n")
            tokens = re.findall(PAT, part)
            for j, token in enumerate(tokens):
                raw_bytes = token.encode("utf-8")
                token = [bytes([b]) for b in token.encode("utf-8")]  # 这个和tuple(token)不同
                if raw_bytes in self.vocab_reverse:
                    encoded_tokens.append(self.vocab_reverse[raw_bytes])
                    continue
                bytes_list = list(token)
                for merge_pair in self.merges:
                    a, b = merge_pair
                    new_seq = []
                    i = 0
                    while i < len(bytes_list):
                        if i + 1 < len(bytes_list) and bytes_list[i] == a and bytes_list[i+1] == b:
                            new_seq.append(a + b)
                            i += 2
                        else:
                            new_seq.append(bytes_list[i])
                            i += 1
                    bytes_list = new_seq
                for token_seq in bytes_list:
                    encoded_tokens.append(self.vocab_reverse[token_seq])
        return encoded_tokens
        

    def encode_iterable(self, iterable):
        # vocab_reverse = {v: k for k, v in self.vocab.items()}
        # encoded_tokens = []
        # for j, token in enumerate(iterable):
        #     raw_bytes = token.encode("utf-8")
        #     token = [bytes([b]) for b in raw_bytes]  # 这个和tuple(token)不同
        #     if raw_bytes in vocab_reverse:
        #         encoded_tokens.append(vocab_reverse[raw_bytes])
        #         continue
        #     bytes_list = list(token)
        #     for merge_pair in self.merges:
        #         a, b = merge_pair
        #         new_seq = []
        #         i = 0
        #         while i < len(bytes_list):
        #             if i + 1 < len(bytes_list) and bytes_list[i] == a and bytes_list[i+1] == b:
        #                 new_seq.append(a + b)
        #                 i += 2
        #             else:
        #                 new_seq.append(bytes_list[i])
        #                 i += 1
        #         bytes_list = new_seq
        #     for token_seq in bytes_list:
        #         encoded_tokens.append(vocab_reverse[token_seq])
        # return encoded_tokens
        encoded_tokens = []
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True) # 这个地方一定要注意啊，正则匹配最长匹配原则
            pattern = "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
            for line in iterable:
                line = line.replace("\r\n", "\n").replace("\r", "\n")
                parts = re.split(pattern, line)
                for part in parts:
                    if part == "": 
                        continue
                    if part in self.special_tokens:
                        encoded_tokens.append(self.vocab_reverse[part.encode("utf-8")])
                        continue
                    for token in re.findall(PAT, part):
                        raw_bytes = token.encode("utf-8")
                        token = [bytes([b]) for b in token.encode("utf-8")]  # 这个和tuple(token)不同
                        if raw_bytes in self.vocab_reverse:
                            encoded_tokens.append(self.vocab_reverse[raw_bytes])
                            continue
                        bytes_list = list(token)
                        for merge_pair in self.merges:
                            a, b = merge_pair
                            new_seq = []
                            i = 0
                            while i < len(bytes_list):
                                if i + 1 < len(bytes_list) and bytes_list[i] == a and bytes_list[i+1] == b:
                                    new_seq.append(a + b)
                                    i += 2
                                else:
                                    new_seq.append(bytes_list[i])
                                    i += 1
                            bytes_list = new_seq
                        for token_seq in bytes_list:
                            encoded_tokens.append(self.vocab_reverse[token_seq])
        else:
            for line in iterable:
                line = line.replace("\r\n", "\n").replace("\r", "\n")
                for token in re.findall(PAT, line):
                    raw_bytes = token.encode("utf-8")
                    token = [bytes([b]) for b in token.encode("utf-8")]  # 这个和tuple(token)不同
                    if raw_bytes in self.vocab_reverse:
                        encoded_tokens.append(self.vocab_reverse[raw_bytes])
                        continue
                    bytes_list = list(token)
                    for merge_pair in self.merges:
                        a, b = merge_pair
                        new_seq = []
                        i = 0
                        while i < len(bytes_list):
                            if i + 1 < len(bytes_list) and bytes_list[i] == a and bytes_list[i+1] == b:
                                new_seq.append(a + b)
                                i += 2
                            else:
                                new_seq.append(bytes_list[i])
                                i += 1
                        bytes_list = new_seq
                    for token_seq in bytes_list:
                        encoded_tokens.append(self.vocab_reverse[token_seq])
        return encoded_tokens

    def decode(self, tokens: list[int]) -> str:
        decoded_text = b""
        for token in tokens:
            decoded_text += self.vocab[token]
        return decoded_text.decode("utf-8", errors="replace")
    
if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"C:\Users\Dy.ming\Desktop\CS336\assignment1-basics")

    from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
    VOCAB_PATH = 'tests/fixtures/gpt2_vocab.json'
    MERGES_PATH = 'tests/fixtures/gpt2_merges.txt' 
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )
    # test_string = "Héllò hôw are ü? 🙃"
    # encoded_ids = tokenizer.encode(test_string)
    # tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # # Ensure the special <|endoftext|> token is preserved
    # assert tokenized_string.count("<|endoftext|>") == 3

    # decoded_string = tokenizer.decode(encoded_ids)
    # assert test_string == decoded_string
    import tiktoken
    ref = tiktoken.get_encoding("gpt2")
    # 你的 tokenizer
    print("tiktoken:", ref.encode("Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>", allowed_special={"<|endoftext|>"}))
    print("yours:   ", tokenizer.encode("Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"))

    print("vocab[127]:", tokenizer.vocab[127])
    print("tiktoken decode([127]):", ref.decode([127]))