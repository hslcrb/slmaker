class Tokenizer:
    """ 🚀 Byte-level Latent Tokenizer / 바이트 단위 잠재 토크나이저
    Handles UTF-8 bytes directly to support English, Korean, and Code.
    """
    def __init__(self, text=None):
        # We use a fixed 256 byte vocab + special tokens / 256개 바이트 + 특수 토큰 고정 어휘 사전
        self.vocab_size = 256 + 1 # +1 for <|endoftext|>
        self.eot_token = 256

    def encode(self, s):
        """ String to list of byte integers with EOT support / EOT 지원을 포함한 문자열-바이트 정수 변환 """
        res = []
        # Split by <|endoftext|> marker / 마커를 기준으로 분할
        parts = s.split("<|endoftext|>")
        for i, part in enumerate(parts):
            res.extend(list(part.encode('utf-8')))
            if i < len(parts) - 1:
                res.append(self.eot_token)
        return res

    def decode(self, l):
        """ List of byte integers to string / 바이트 정수 리스트를 문자열로 변환 """
        # Filter out EOT for clean decoding / 깨끗한 디코딩을 위해 EOT 필터링
        bytes_list = bytes([b for b in l if b < 256])
        return bytes_list.decode('utf-8', errors='replace')

if __name__ == "__main__":
    # Test / 테스트
    t = Tokenizer()
    test_str = "Hello, slmaker! 안녕하세요! def code(): pass"
    encoded = t.encode(test_str)
    decoded = t.decode(encoded)
    print(f"Original: {test_str}")
    print(f"Encoded: {encoded[:10]}...")
    print(f"Decoded: {decoded}")
    assert test_str == decoded
    print("Tokenizer Upgrade Verified. / 토크나이저 고도화 검증 완료.")
