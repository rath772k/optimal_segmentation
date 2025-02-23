from transformers import GPT2Tokenizer
import tiktoken 
import regex as re

class OptimalGPT2Tokenizer(GPT2Tokenizer):
    """
    To use the tokenizer with HF tokenizer, `bpe` method must be overridden.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # Precompute vocab dictionaries
        self.vocab_dict = self.get_vocab()
        self.inverse_vocab = {v: k for k, v in self.vocab_dict.items()}
        
        # Build the trie once during initialization
        self.root = self.build_trie()
        print(f"Len of Vocab Initialized is {len(self.vocab_dict)}")

    def optimal_segmentation(self, tokens):
        n = len(tokens)
        dp = [float('inf')] * (n + 1)
        dp[-1] = 0  # dp[n] = dp[-1] is the base case for dynamic programming
        par = [i-1 for i in range(n)]  # To keep track of the partition positions, initialized assuming worst case segmentation length
        
        for i in range(n):
            cur_node = self.root
            for j in range(i, -1, -1):
                token = tokens[j]
                cur_node = cur_node.get(token)
                if cur_node is None:
                    break
                if cur_node.get("end", False):
                    cost = dp[j - 1] + 1
                    if cost < dp[i]:
                        dp[i] = cost
                        par[i] = j - 1

        # Reconstruct the tokens from parent pointers
        new_tokens = []
        cur = n - 1
        while cur >= 0:
            start, end = par[cur] + 1, cur + 1
            token_str = "".join(tokens[start: end])
            token_id = self.vocab_dict.get(token_str)
            if token_id is None:
                # Handle unknown token (should not happen if trie is built correctly)
                raise ValueError(f"Token '{token_str}' not found in vocabulary.")
            new_tokens.append(token_id)
            cur = par[cur]
        new_tokens.reverse()
        return new_tokens

    def insert(self, root, word):
        cur = root
        for c in word:
            cur[c] = cur.get(c, {})
            cur = cur[c]
        cur["end"] = True

    def build_trie(self):
        root = {}
        for word in self.vocab_dict.keys():
            self.insert(root, word[::-1])
        return root

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)

        word_pieces = self.optimal_segmentation(word)
        # Convert token IDs back to token strings
        tokens = [self.inverse_vocab[piece] for piece in word_pieces]
        word = " ".join(tokens)
        self.cache[token] = word
        return word



class OptimalTiktokenTokenizer(OptimalGPT2Tokenizer):
    def __init__(self, model_name: str, *args, **kwargs):
        self.cache = {} # since we aren't using parent's init      
        enc = tiktoken.encoding_for_model(model_name)
        self.pat = re.compile(enc._pat_str)        
        self.b2u = self.bytes_to_unicode()
        self.vocab_dict = self.convert_vocab_to_hf(enc._mergeable_ranks)
        self.inverse_vocab = {v: k for k, v in self.vocab_dict.items()}

        # Build the trie once during initialization
        self.root = self.build_trie()
        print(f"Len of Vocab Initialized is {len(self.vocab_dict)}")
        
    def convert_vocab_to_hf(self,tiktoken_vocab):
        # Create the Hugging Face-compatible vocabulary
        hf_vocab = {}
        for token_bytes, token_id in tiktoken_vocab.items():
            if isinstance(token_bytes, bytes):
                token_str = self.byte_seq_to_unicode_string(token_bytes)
            else:
                # If token_bytes is already a string (unlikely), use it directly
                token_str = token_bytes
            hf_vocab[token_str] = token_id
        return hf_vocab

    def bytes_to_unicode(self):
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a signficant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        And avoids mapping to whitespace/control characters the bpe code barfs on.
        """
        bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
        
    def byte_seq_to_unicode_string(self,byte_seq):
        return ''.join([self.b2u[b] for b in byte_seq])
    