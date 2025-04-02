### Quick Start: Try Default Search Tool on HotpotQA
#### 1. Install `FlagEmbedding` and `faiss`
```bash
pip3 install FlagEmbedding
pip3 install faiss-cpu
```

#### 2. Download and preprocess HotpotQA dataset
```bash
# Create data directory
mkdir -p data/hotpotqa

# Run the preprocessing script
python examples/data_preprocess/hotpotqa.py --local_dir ./data/hotpotqa
```

This script will:
- Download the HotpotQA dataset directly from the source
- Process the data into the format required by Agent-R1
- Save the processed data as train.parquet and validation.parquet in the specified directory

#### 3. Build hotpotqa search index
```bash
# Download the corpus file (gzipped)
mkdir -p data/corpus/hotpotqa
wget https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus.jsonl.gz -O data/corpus/hotpotqa/corpus.jsonl.gz

# Extract the gzipped file
gunzip -c data/corpus/hotpotqa/corpus.jsonl.gz > data/corpus/hotpotqa/hpqa_corpus.jsonl

# Process the corpus and build the search index
cd scripts/hotpotqa_search
python process_hotpotqa.py
```

This script will:
- Load the corpus data
- Generate embeddings using the BAAI/bge-large-en-v1.5 model
- Build a FAISS index for efficient similarity search
- Save the embeddings and index files in the data/corpus/hotpotqa directory

#### 4. Run PPO/REINFORCE++/GRPO training with Qwen2.5-1.5B-Instruct
```bash
# Run the PPO training script
bash run_ppo.sh
# Run the REINFORCE++ training script
bash run_rpp.sh
# Run the GRPO training script
bash run_grpo.sh
```
