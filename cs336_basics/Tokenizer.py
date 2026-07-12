import json
import regex as re
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.merge_lookup = {}
        for rank, pair in enumerate(merges):
            a, b = pair
            self.merge_lookup.setdefault(a, []).append((a, b, rank))

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
                    is_end = False
                    while not is_end:
                        is_end = True
                        # 第一遍扫描：找全局 rank 最小的可合并位置
                        best_rank = float('inf')
                        best_i = -1
                        best_pair = None
                        for i in range(len(bytes_list) - 1):
                            if bytes_list[i] in self.merge_lookup:
                                for a, b, rank in self.merge_lookup[bytes_list[i]]:
                                    if bytes_list[i+1] == b and rank < best_rank:
                                        best_rank = rank
                                        best_i = i
                                        best_pair = (a, b)
                                        break  # 因为列表已按 rank 排序，第一个匹配就是该位置的最优
                        if best_pair is not None:
                            is_end = False
                            a, b = best_pair
                            new_seq = []
                            i = 0
                            while i < len(bytes_list):
                                if i == best_i:
                                    new_seq.append(a + b)
                                    i += 2
                                else:
                                    new_seq.append(bytes_list[i])
                                    i += 1
                            bytes_list = new_seq
                    #     new_seq = []      
                    #     i = 0
                    #     while i < len(bytes_list):
                    #         if i + 1 < len(bytes_list) and bytes_list[i] == a and bytes_list[i+1] == b:
                    #             new_seq.append(a + b)
                    #             i += 2
                    #         else:
                    #             new_seq.append(bytes_list[i])
                    #             i += 1
                    #     bytes_list = new_seq
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