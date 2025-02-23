class OptimalBPETokenizer:
    def __init__(self, vocab, merges, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        self.inv_vocab = {v:k for k, v in self.vocab.items()}
        self.root = self.build_trie()
        self.merges = merges
        self.vocab_size = len(self.vocab)

    def encode_greedy(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            replacement_pair = min(
                [(cur, nxt) for cur, nxt in zip(tokens, tokens[1:])],
                key=lambda pair: self.merges.get(pair, 1e9),
            )
            if replacement_pair not in self.merges:
                break
            replacement_idx = self.merges[replacement_pair]
            tokens = self.merge(tokens, replacement_pair, replacement_idx)
        return tokens
    
    def encode_optimal(self, text):
        tokens = bytes(list(text.encode("utf-8")))
        n = len(tokens)
        dp = [i for i in range(1, n+1)] + [0]
        par = [i-1 for i in range(n)]
        for i in range(n):
            cur = self.root
            for j in range(i, -1, -1):
                cur = cur.get(tokens[j])
                if cur is None:
                    break
                if cur.get("end", False):
                    if dp[j-1] + 1 < dp[i]:
                        dp[i] = dp[j-1] +  1
                        par[i] = j-1
        new_tokens = []
        cur = n-1
        while cur != -1:
            new_tokens.append(self.inv_vocab[tokens[par[cur]+1: cur+1]])
            cur = par[cur]
        new_tokens.reverse()
        return new_tokens

    def decode(self, tokens):
        byte_string = b"".join(self.vocab[token] for token in tokens)
        return byte_string.decode("utf-8", errors="replace")

    @classmethod
    def build_tokenizer(cls, filepath, target_vocab_size):
        with open(filepath, "r") as f:
            data = f.read()
        unicode_data = list(data.encode("utf-8"))
        vocab = {idx: bytes([idx]) for idx in range(256)}
        merges = {}

        for _ in range(target_vocab_size - 256):
            if len(unicode_data) == 1:
                break
            freq = cls.get_freq(unicode_data)
            merge_pair = max(freq, key=freq.get)

            new_idx = len(vocab)
            merges[merge_pair] = new_idx

            byte_pair = vocab[merge_pair[0]] + vocab[merge_pair[1]]
            vocab[new_idx] = byte_pair

            unicode_data = cls.merge(unicode_data, merge_pair, new_idx)

        return cls(vocab, merges)

    @staticmethod
    def merge(tokens, pair, idx):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    @staticmethod
    def get_freq(data):
        from collections import defaultdict

        freq = defaultdict(int)
        for cur, nxt in zip(data, data[1:]):
            freq[(cur, nxt)] += 1
        return freq
    
    def build_trie(self):
        root = {}
        def insert(word):
            cur = root
            for c in word:
                cur[c] = cur.get(c, {})
                cur = cur[c]
            cur["end"] = True
        for word in self.vocab.values():
            insert(word[::-1])
        
        return root