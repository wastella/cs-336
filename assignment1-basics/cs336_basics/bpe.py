import regex
import re
import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {num+len(special_tokens): bytes([num]) for num in range(256)}
    for i in range(len(special_tokens)):
       vocab[i] = bytes(special_tokens[i], "utf-8")
    merges = []

    freq = {}

    with open(input_path, encoding="utf-8") as f:
        full_text = f.read()

        special_tokens = [re.escape(tok) for tok in special_tokens]

        chunks = re.split("|".join(special_tokens), full_text)

        for chunk in chunks:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            corpus = regex.findall(PAT, chunk)

            sub_freq = {}
            for tok in corpus:
                tok_as_byte = tok.encode("utf-8")
                tup = tuple(bytes([sub_byte]) for sub_byte in tok_as_byte)
                if tup in sub_freq:
                    sub_freq[tup] += 1
                else:
                    sub_freq[tup] = 1

            for entry in sub_freq:
                if entry in freq:
                    freq[entry] += sub_freq[entry]
                else:
                    freq[entry] = sub_freq[entry]

    while len(vocab) < vocab_size:
        pair_freq = {}
        for tok in freq:
            for i in range(len(tok)-1):
                pair = (tok[i], tok[i+1])
                if pair in pair_freq:
                    pair_freq[pair] += freq[tok]
                else:
                    pair_freq[pair] = freq[tok]

        greatest_n = max(pair_freq, key=lambda x: (pair_freq.get(x), x))
        vocab[len(vocab)] = greatest_n[0] + greatest_n[1]
        merges.append(greatest_n)

        for tok in list(freq):
            new_tok = []
            trailing = False
            for i in range(len(tok)):
                if i == len(tok)-1:
                    if trailing:
                        trailing = False
                    else:
                        new_tok.append(tok[i])
                else:
                    forward_pair = (tok[i], tok[i+1])
                    if forward_pair == greatest_n:
                        new_tok.append(tok[i] + tok[i+1])
                        trailing = True
                    elif trailing:
                        trailing = False
                    else:
                        new_tok.append(tok[i])

            val = freq[tok]
            freq.pop(tok)
            freq[tuple(new_tok)] = val

    #print(vocab)
    return vocab, merges
