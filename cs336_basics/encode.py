import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.tokenizer import Tokenizer
import argparse

datasets = {
    'tinystories': {
        'train': '../data/TinyStoriesV2-GPT4-train.txt',
        'val': '../data/TinyStoriesV2-GPT4-valid.txt',
        'vocab_filepath': 'results/tinystories/vocab.json',
        'merges_filepath': 'results/tinystories/merges.json',
        'special_tokens': ['<|endoftext|>'],
        'output_dir': 'results/tinystories'
    },
    'owt': {
        'train': '/data/a1-basics/owt_train.txt',
        'val': '/data/a1-basics/owt_valid.txt',
        'vocab_filepath': '/home/c-puhengli/assignment1-basics/cs336_basics/results/owt/vocab.json',
        'merges_filepath': '/home/c-puhengli/assignment1-basics/cs336_basics/results/owt/merges.json',
        'special_tokens': ['<|endoftext|>'],
        'output_dir': 'results/owt'
    }
}

def encode_chunk(args):
    chunk, tokenizer = args
    return tokenizer.encode(chunk)

def parallel_encode(text: str, tokenizer, num_chunks: int = 32) -> list[int]:
    length = len(text)
    chunk_size = length // num_chunks
    chunks = [text[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    
    # merge last chunk
    chunks[-1] += text[num_chunks*chunk_size:]
    
    chunk_tokenizer_pairs = [(chunk, tokenizer) for chunk in chunks]
    
    with ProcessPoolExecutor() as executor:
        encoded_parts = list(tqdm(
            executor.map(encode_chunk, chunk_tokenizer_pairs), 
            total=num_chunks, 
            desc="Encoding chunks"
        ))
    
    return [token for part in encoded_parts for token in part]

def process_dataset(dataset_name, overall_progress=None):
    dataset = datasets[dataset_name]

    tokenizer = Tokenizer.from_files(
        vocab_filepath=dataset['vocab_filepath'],
        merges_filepath=dataset['merges_filepath'],
        special_tokens=dataset['special_tokens']
    )
    
    # encode
    splits = ['train', 'val']
    for split_idx, split in enumerate(splits):
        if overall_progress:
            overall_progress.set_description(f"Dataset: {dataset_name} | Split: {split}")
        
        with open(dataset[split], 'r', encoding='utf-8') as f:
            text = f.read()
            encoded = parallel_encode(text, tokenizer)
        
        # memmap
        output_path = os.path.join(dataset['output_dir'], f"{split}.bin")
        arr = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(len(encoded),))
        arr[:] = np.array(encoded, dtype=np.uint16)
        arr.flush()
        
        if overall_progress:
            overall_progress.update(1)  
            
        print(f"Saved dataset {dataset_name}/{split} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizing")
    parser.add_argument('--datasets', nargs='+', choices=list(datasets.keys()) + ['all'], 
                        default=['all'], help='Datasets')
    
    args = parser.parse_args()
    
    datasets_to_process = list(datasets.keys()) if 'all' in args.datasets else args.datasets
    
    # train and val
    total_tasks = sum(2 for dataset_name in datasets_to_process) 
    
    progress = tqdm(total=total_tasks, desc="Progress", position=0)
    
    for dataset_name in datasets_to_process:
        try:
            process_dataset(dataset_name, progress)
        except Exception as e:
            print(f"Encounter error when processing dataset {dataset_name}: {e}")
            progress.update(2)  
    
    progress.close()
    print("Completed!")