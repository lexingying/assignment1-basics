import regex as re
from typing import Iterable, List, Tuple
import json
import array
from functools import lru_cache

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
REGEX_PAT = re.compile(PAT)

class Tokenizer:
    def __init__(self, vocab: dict, merges: list, special_tokens: list = None):
        self.int_to_byte = {}
        for token_id, byte_list in vocab.items():
            self.int_to_byte[int(token_id)] = bytes(byte_list)
        
        self.byte_to_int = {v: k for k, v in self.int_to_byte.items()}
        
        # Ensure completeness
        for i in range(256):
            byte = bytes([i])
            if byte not in self.byte_to_int:
                self.byte_to_int[byte] = len(self.byte_to_int)
                self.int_to_byte[len(self.int_to_byte)] = byte
        
        # BPE merge table
        self.bpe_ranks = {}
        for rank, merge_pair in enumerate(merges):
            if len(merge_pair) == 2:
                a_bytes = bytes(merge_pair[0])
                b_bytes = bytes(merge_pair[1])
                
                # token IDs
                if a_bytes in self.byte_to_int:
                    a_id = self.byte_to_int[a_bytes]
                else:
                    a_id = len(self.byte_to_int)
                    self.byte_to_int[a_bytes] = a_id
                    self.int_to_byte[a_id] = a_bytes
                
                if b_bytes in self.byte_to_int:
                    b_id = self.byte_to_int[b_bytes]
                else:
                    b_id = len(self.byte_to_int)
                    self.byte_to_int[b_bytes] = b_id
                    self.int_to_byte[b_id] = b_bytes
                
                # merged token ID
                merged_bytes = a_bytes + b_bytes
                if merged_bytes in self.byte_to_int:
                    merged_id = self.byte_to_int[merged_bytes]
                else:
                    merged_id = len(self.byte_to_int)
                    self.byte_to_int[merged_bytes] = merged_id
                    self.int_to_byte[merged_id] = merged_bytes
                
                # BPE rank for pair
                pair = (a_id, b_id)
                self.bpe_ranks[pair] = (rank, merged_id)

        # Special tokens
        self.special_tokens = {}
        self.special_tokens_regex = None
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.byte_to_int:
                    token_id = len(self.byte_to_int)
                    self.byte_to_int[token_bytes] = token_id
                    self.int_to_byte[token_id] = token_bytes
                self.special_tokens[token] = self.byte_to_int[token_bytes]
            
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            self.special_tokens_regex = re.compile(special_pattern)
            
        self._byte_cache = {i: bytes([i]) for i in range(256)}
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
    
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges = json.load(f)
            
        return cls(vocab, merges, special_tokens)

    @property
    def vocab_size(self):
        return len(self.int_to_byte)
    
    # encode token with caching
    @lru_cache(maxsize=8192)
    def encode_token_cache(self, token: str) -> Tuple[int, ...]:
        token_bytes = token.encode("utf-8")
        if token_bytes in self.byte_to_int:
            return (self.byte_to_int[token_bytes],)
        
        # get token IDs
        ids = array.array('I', [self.byte_to_int[self._byte_cache.get(b, bytes([b]))] for b in token_bytes])

        if len(ids) <= 1:
            return tuple(ids)
        
        # apply merge
        ids = self.bpe_merge(ids)
        return tuple(ids)
    
    def bpe_merge(self, ids: array.array) -> List[int]:
        if len(ids) <= 1:
            return list(ids)

        ids_list = list(ids)
        
        # merge positions
        positions = list(range(len(ids_list) - 1))
        
        while positions:
            # the best merge
            best_rank = float('inf')
            best_pos = -1
            best_id = -1
            
            for pos in positions[:]: 
                if pos >= len(ids_list) - 1:
                    positions.remove(pos)
                    continue
                
                pair = (ids_list[pos], ids_list[pos + 1])
                if pair in self.bpe_ranks:
                    rank, merged_id = self.bpe_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pos = pos
                        best_id = merged_id
            
            # if no more merges
            if best_pos == -1:
                break
                
            # merge
            ids_list[best_pos] = best_id
            ids_list.pop(best_pos + 1)
            
            # update positions
            positions = [p if p < best_pos else p - 1 for p in positions if p != best_pos]
            
            # update merge positions
            if best_pos > 0 and best_pos - 1 not in positions:
                positions.append(best_pos - 1)
            if best_pos < len(ids_list) - 1 and best_pos not in positions:
                positions.append(best_pos)
        
        return ids_list
    
    def encode(self, text: str, progress_bar: bool = False) -> List[int]:
        if not text:
            return []
        
        result = []

        if self.special_tokens_regex:
            chunks = self.special_tokens_regex.split(text)
            for chunk in chunks:
                if not chunk:
                    continue
                    
                if chunk in self.special_tokens:
                    result.append(self.special_tokens[chunk])
                else:
                    for token in REGEX_PAT.findall(chunk):
                        result.extend(self.encode_token_cache(token))
        else:
            for token in REGEX_PAT.findall(text):
                result.extend(self.encode_token_cache(token))
        
        return result
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]
    
    def encode_iterable(self, texts: Iterable[str]) -> Iterable[int]:
        for text in texts:
            yield from self.encode(text)
    
    def decode(self, ids: List[int]) -> str:
        buffer = bytearray()
        for i in ids:
            if i in self.int_to_byte:
                buffer.extend(self.int_to_byte[i])
            else:
                buffer.extend(b'?')
        return buffer.decode("utf-8", errors="replace")