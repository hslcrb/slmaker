class Tokenizer:
    """ Memory-efficient Character-level Tokenizer / 메모리 효율적인 문자 단위 토크나이저 """
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s):
        """ String to list of integers / 문자열을 정수 리스트로 변환 """
        return [self.stoi[c] for c in s]

    def decode(self, l):
        """ List of integers to string / 정수 리스트를 문자열로 변환 """
        return ''.join([self.itos[i] for i in l])
