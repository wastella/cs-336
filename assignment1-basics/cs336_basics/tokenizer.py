import regex
import re
import pickle

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab

        self.vocab_reversed = {}

        for key in self.vocab:
            self.vocab_reversed[self.vocab[key]] = key

        self.merges = merges
        self.is_special_tokens = (special_tokens != None)
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f, open(merges_filepath, "rb") as mf:
            vocab = pickle.load(f)
            merges = pickle.load(mf)
            return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text):
        final_tokens = []

        if not self.is_special_tokens:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pretokenized = regex.findall(PAT, text)

            for i in range(len(pretokenized)):
                pretokenized[i] = pretokenized[i].encode("utf-8")

            pretok_lists = []

            for pretok in pretokenized:
                byte_list = [bytes([sub_byte]) for sub_byte in pretok]
                for merge in self.merges:
                    i = 0
                    while i < len(byte_list) - 1:

                        pair = (byte_list[i], byte_list[i+1])
                        if pair == merge:
                            byte_list[i] = pair[0] + pair[1]
                            byte_list.pop(i+1)
                        i += 1
                    #byte_list = byte_list_new

                converted_byte_list = [self.vocab_reversed[key] for key in byte_list]

                pretok_lists.append(converted_byte_list)

            for l in pretok_lists:
                final_tokens += l

        else:
            special_tokens = [re.escape(tok) for tok in self.special_tokens]
            special_tokens.sort(key=len, reverse=True)
            print(special_tokens)
            chunks = re.split("(" + "|".join(special_tokens) + ")", text)

            for chunk in chunks:
                if chunk in self.special_tokens:
                    final_tokens.append(self.vocab_reversed[chunk.encode("utf-8")])

                else:
                    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                    pretokenized = regex.findall(PAT, chunk)

                    for i in range(len(pretokenized)):
                        pretokenized[i] = pretokenized[i].encode("utf-8")

                    pretok_lists = []

                    for pretok in pretokenized:
                        byte_list = [bytes([sub_byte]) for sub_byte in pretok]

                        for merge in self.merges:
                            i = 0
                            while i < len(byte_list) - 1:
                                pair = (byte_list[i], byte_list[i+1])
                                if pair == merge:
                                    byte_list[i] = pair[0] + pair[1]
                                    byte_list.pop(i+1)
                                i += 1
                            #byte_list = byte_list_new

                        converted_byte_list = [self.vocab_reversed[key] for key in byte_list]
                        pretok_lists.append(converted_byte_list)

                    for l in pretok_lists:
                        final_tokens  += l


        return final_tokens

    def encode_iterable(self, iterable):
        for text in iterable:
            for id in self.encode(text):
                yield id

    def decode(self, ids):
        decoded = b''
        for token in ids:
            decoded += self.vocab[token]
        return decoded.decode("utf-8", errors="replace")

"""
vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't',
         6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at', 11: b'<|special|>'}
merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]

tokenizer = Tokenizer(vocab, merges)
ids = tokenizer.encode("thetha")
dec_ids = [9, 7, 1, 5, 0, 11, 10, 3]
print(ids)
"""
