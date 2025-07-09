import regex as re
from typing import Iterable, List, Dict, Set, Tuple
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from collections import Counter, defaultdict
import concurrent.futures
import mmap

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    memory_map: mmap.mmap,
    target_segments: int,
    delimiter: bytes
) -> list[int]:
    """
    Determine segment boundaries on an mmap'ed file object.
    Uses memory mapping instead of iterative file.seek() calls for better performance.
    """
    total_size = memory_map.size()
    segment_size = total_size // target_segments

    # Initial guesses for segment boundaries.
    segment_boundaries = [i * segment_size for i in range(target_segments + 1)]
    segment_boundaries[-1] = total_size

    buffer_size = 4096  # Process 4 KB buffers.
    for idx in range(1, len(segment_boundaries) - 1):
        current_position = segment_boundaries[idx]
        # Slide over the memory map in buffers to locate a delimiter.
        while current_position < total_size:
            end_position = min(current_position + buffer_size, total_size)
            buffer_data = memory_map[current_position:end_position]
            if not buffer_data:
                # End-of-file reached.
                segment_boundaries[idx] = total_size
                break
            delimiter_position = buffer_data.find(delimiter)
            if delimiter_position != -1:
                segment_boundaries[idx] = current_position + delimiter_position
                break
            current_position += buffer_size

    return sorted(set(segment_boundaries))


def _extract_subtokens(text: str):
    return Counter(re.findall(PAT, text))

def _handle_segment(segment_text: str, special_tokens: List[str]):
    if special_tokens:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        fragments = re.split(f"({pattern})", segment_text)
        
        subtoken_counter = Counter()
        for fragment in fragments:
            if fragment not in special_tokens:
                subtoken_counter.update(_extract_subtokens(fragment))
    else:
        subtoken_counter = _extract_subtokens(segment_text)
        
    # Convert subtokens into tuple of bytes
    create_byte_tuple = lambda subtoken: tuple(bytes([b]) for b in subtoken.encode("utf-8"))
    subtoken_frequency = {
        create_byte_tuple(subtoken): freq for subtoken, freq in subtoken_counter.items()
    }
    
    return subtoken_frequency

def _process_file_in_segments(input_path: str, num_processes: int, special_tokens: Iterable[str]):
    logging.info(f"Segmenting file at {input_path} with {num_processes} workers")
    
    delimiter_bytes = "<|endoftext|>".encode("utf-8")
    segments = []
    
    with open(input_path, "rb") as file:
        memory_map = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        total_size = memory_map.size()
        
        # Find boundaries
        boundaries = find_chunk_boundaries(memory_map, num_processes, delimiter_bytes)
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            segment_bytes = memory_map[start:end]
            segment_text = segment_bytes.decode("utf-8", errors="ignore")
            segments.append(segment_text)
        memory_map.close()

    logging.info("Starting pre-tokenization")
    subtoken_freq_total = Counter()
    if num_processes == 1:
        for segment in tqdm(segments):
            segment_freq = _handle_segment(segment, list(special_tokens))
            subtoken_freq_total.update(segment_freq)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            special_tokens_list = list(special_tokens)
            future_to_segment = {
                executor.submit(_handle_segment, segment, special_tokens_list): i 
                for i, segment in enumerate(segments)
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_segment), total=len(segments)):
                subtoken_freq_total.update(future.result())
                    
    return subtoken_freq_total

def _merge_byte_tuple(byte_tuple, pair, merged_bytes):
    result = []
    i = 0
    while i < len(byte_tuple):
        if i < len(byte_tuple) - 1 and (byte_tuple[i], byte_tuple[i+1]) == pair:
            result.append(merged_bytes)
            i += 2
        else:
            result.append(byte_tuple[i])
            i += 1
    return tuple(result)

def train_bpe(input_path: str, vocab_size: int, special_tokens: Iterable[str],
              show_progress: bool = False, num_processes: int = 1):

    vocabulary = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocabulary[256+i] = token.encode("utf-8")
    
    subtoken_freq = _process_file_in_segments(input_path, num_processes, special_tokens)
    
    # initialize the pair frequency table
    pair_to_subtokens: Dict[Tuple[int, int], Set[Tuple[bytes, ...]]] = defaultdict(set)
    pair_frequency: Counter = Counter()
    for subtoken, freq in tqdm(subtoken_freq.items(), disable=not show_progress, desc="Building initial pair mappings"):
        for i in range(len(subtoken) - 1):
            pair = (subtoken[i], subtoken[i+1])
            pair_frequency[pair] += freq
            pair_to_subtokens[pair].add(subtoken)
    
    merge_operations = []
    progress_bar = tqdm(total=vocab_size - len(vocabulary), desc="Merging", disable=not show_progress) if show_progress else None
    
    # perform merges
    while len(vocabulary) < vocab_size:
        if not pair_frequency:
            logging.warning("No more pairs to merge.")
            break
        
        # most frequent pair
        best_pair = max(pair_frequency, key=lambda k: (pair_frequency[k], k))
        merge_operations.append(best_pair)
        
        # merge
        new_id = len(vocabulary)
        merged_bytes = b"".join(best_pair)
        vocabulary[new_id] = merged_bytes
        
        # get the affected subtokens
        affected_subtokens = pair_to_subtokens.get(best_pair, set()).copy()
        
        for subtoken in affected_subtokens:
            freq = subtoken_freq[subtoken]
            new_subtoken = _merge_byte_tuple(subtoken, best_pair, merged_bytes)
            if new_subtoken == subtoken:
                continue 
            
            # Update pair counts
            for i in range(len(subtoken) - 1):
                old_pair = (subtoken[i], subtoken[i+1])
                pair_frequency[old_pair] -= freq
                pair_to_subtokens[old_pair].discard(subtoken)
                if pair_frequency[old_pair] <= 0:
                    del pair_frequency[old_pair]
                    if old_pair in pair_to_subtokens:
                        del pair_to_subtokens[old_pair]
            
            for i in range(len(new_subtoken) - 1):
                new_pair = (new_subtoken[i], new_subtoken[i+1])
                pair_frequency[new_pair] = pair_frequency.get(new_pair, 0) + freq
                if new_pair not in pair_to_subtokens:
                    pair_to_subtokens[new_pair] = set()
                pair_to_subtokens[new_pair].add(new_subtoken)
            
            # Update the subtoken frequency table
            subtoken_freq[new_subtoken] = subtoken_freq.get(new_subtoken, 0) + freq
            del subtoken_freq[subtoken]
        
        if best_pair in pair_frequency:
            del pair_frequency[best_pair]
        if best_pair in pair_to_subtokens:
            del pair_to_subtokens[best_pair]
        
        if show_progress and progress_bar:
            progress_bar.update(1)
    
    if show_progress and progress_bar:
        progress_bar.close()

    return vocabulary, merge_operations