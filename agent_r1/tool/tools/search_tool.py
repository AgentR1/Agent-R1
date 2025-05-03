"""
Search tool implementation for simulating internet searches
"""

import time
import random
from typing import Dict, List, Any, Optional
import os

from agent_r1.tool.tool_base import Tool

# from txtai.embeddings import Embeddings
import faiss
from FlagEmbedding import FlagAutoModel
import json

class SearchTool(Tool):
    """
    Tool for simulating internet searches using the NeuML/txtai-wikipedia model
    """
    
    def __init__(self):
        """
        Initialize the search tool
        
        Args:
            search_db: Custom search database, if None, use default
        """
        name = "search"
        description = "Search for information on the internet using Wikipedia as a knowledge source."
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                # "limit": {
                #     "type": "integer",
                #     "description": "Maximum number of results to return (default: 5)"
                # }
            },
            "required": ["query"]
        }
        
        super().__init__(name, description, parameters)
        print("[DEBUG] EMBEDDINGS LOADING")
        
        # Get the absolute path to the data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(current_dir, "../../../data/corpus/hotpotqa"))
        
        # Load index and corpus using absolute paths
        self.index = faiss.read_index(os.path.join(data_dir, "index.bin"))
        self.model = FlagAutoModel.from_finetuned(
            'BAAI/bge-large-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            devices="cpu",   # if not specified, will use all available gpus or cpu when no gpu available
        )
        self.corpus = []
        with open(os.path.join(data_dir, "hpqa_corpus.jsonl"), "r") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                self.corpus.append(data['title'] + " " + data["text"])
        print("[DEBUG] EMBEDDINGS LOADING END")

    
    def execute(self, args: Dict) -> str:
        """
        Execute search query
        
        Args:
            args: Tool parameters, containing:
                - "query": search query string
                - "limit": optional int to limit number of results
            
        Returns:
            Formatted search results
        """
        query = args["query"]
        embedding = self.model.encode_queries([query])
        dist, ids = self.index.search(embedding, 5)  # ids: 1*5
        results_str = self._format_results(ids[0])
        return results_str
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        queries = [x["query"] for x in args_list]
        embeddings = self.model.encode_queries(queries)
        dist, ids = self.index.search(embeddings, 5) # ids: b*5
        results_str = [self._format_results(ids[i]) for i in range(len(ids))]
        return results_str

    def _format_results(self, results: List) -> str:
        """
        Format search results for better readability
        
        Args:
            results: List of search result List
            
        Returns:
            Formatted results as a string
        """
        results_list = []
        
        for i, result in enumerate(results):
            results_list.append(self.corpus[result])
        
        return json.dumps({"results": results_list})
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # valid tool call
        if "results" in result:
            return 0.1
        else:
            return 0.0