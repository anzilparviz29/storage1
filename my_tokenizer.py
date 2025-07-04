import json
import re

class GPT2Tokenizer:
    def __init__(self, vocab_file, merges_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v:k for k,v in self.encoder.items()}

        with open(merges_file, 'r', encoding='utf-8') as f:
            merges = f.read().split('\n')[1:-1]
        self.bpe_ranks = dict()
        for i, merge in enumerate(merges):
            self.bpe_ranks[tuple(merge.split())] = i

        self.cache = dict()

        # Regex splits words and punctuation
        self.pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\w\s]+", re.IGNORECASE)

    def get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = list(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        while True:
            min_pair = None
            min_rank = float('inf')
            for pair in pairs:
                if pair in self.bpe_ranks and self.bpe_ranks[pair] < min_rank:
                    min_rank = self.bpe_ranks[pair]
                    min_pair = pair
            if min_pair is None:
                break
            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if i < len(word)-1 and word[i] == first and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word_str = ' '.join(word)
        self.cache[token] = word_str
        return word_str

    def encode(self, text):
        if not text:
            # Return the token ID for '' or fallback to unknown token if not present
            return [self.encoder.get("", self.encoder.get("unk", 50256))]
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token_bpe = self.bpe(token).split(' ')
            bpe_tokens.extend([self.encoder.get(t, self.encoder.get('unk', 50256)) for t in token_bpe])
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder.get(t, '') for t in tokens])
        # Optional: Replace GPT-2 space symbol 'Ġ' with space for readability
        text = text.replace('Ġ', ' ').strip()
        return text
