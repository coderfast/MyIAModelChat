"""MoE (Mixture of Experts) dataset preparation.

This module assigns expert labels to tokens based on their content type:
- Expert 0: General conversation (no special tokens)
- Expert 1: Thinking/reasoning (<thinking>...</thinking>)
- Expert 2: Tool calls (<tool_call>...</tool_call>)
- Expert 3: Observations (<observation>...</observation>)
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MoELabelConfig:
    """Configuration for MoE expert labeling."""
    num_experts: int = 4
    # Expert indices
    expert_conversation: int = 0  # General chat
    expert_thinking: int = 1      # Reasoning/thinking
    expert_tool_call: int = 2     # Tool use
    expert_observation: int = 3   # Observations/results


class MoEDataProcessor:
    """Processes tokenized data to add expert labels for MoE training.
    
    Each token in the dataset gets an expert label based on its context:
    - Tokens outside special blocks → Expert 0 (conversation)
    - Tokens inside <thinking>...</thinking> → Expert 1 (thinking)
    - Tokens inside <tool_call>...</tool_call> → Expert 2 (tool call)
    - Tokens inside <observation>...</observation> → Expert 3 (observation)
    """
    
    def __init__(self, tokenizer, config: Optional[MoELabelConfig] = None):
        """Initialize MoE data processor.
        
        Args:
            tokenizer: Tokenizer instance with special token methods
            config: MoE labeling configuration
        """
        self.tokenizer = tokenizer
        self.config = config or MoELabelConfig()
        
        # Get special token IDs
        self.thinking_id = tokenizer.get_thinking_index()
        self.thinking_end_id = tokenizer.get_thinking_end_index()
        self.tool_call_id = getattr(tokenizer, 'get_tool_call_index', lambda: -1)()
        self.tool_call_end_id = getattr(tokenizer, 'get_tool_call_end_index', lambda: -1)()
        # Use <|tool_result|> prefix (no closing tag; runs until <|end|>/<|assistant|>)
        self.observation_id = getattr(tokenizer, 'get_tool_result_index', lambda: -1)()
        self.observation_end_id = getattr(tokenizer, 'get_end_index', lambda: -1)()
        
        logger.info(f"MoEDataProcessor initialized with {self.config.num_experts} experts")
    
    def assign_expert_labels(self, token_ids: List[int]) -> List[int]:
        """Assign expert labels to a sequence of tokens.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of expert labels (same length as token_ids)
        """
        expert_labels = [self.config.expert_conversation] * len(token_ids)
        
        # Track which block we're in
        in_thinking = False
        in_tool_call = False
        in_observation = False
        
        for i, token_id in enumerate(token_ids):
            # Check for block transitions
            if token_id == self.thinking_id:
                in_thinking = True
                expert_labels[i] = self.config.expert_thinking
            elif token_id == self.thinking_end_id:
                expert_labels[i] = self.config.expert_thinking
                in_thinking = False
            elif token_id == self.tool_call_id:
                in_tool_call = True
                expert_labels[i] = self.config.expert_tool_call
            elif token_id == self.tool_call_end_id:
                expert_labels[i] = self.config.expert_tool_call
                in_tool_call = False
            elif token_id == self.observation_id:
                in_observation = True
                expert_labels[i] = self.config.expert_observation
            elif token_id == self.observation_end_id:
                expert_labels[i] = self.config.expert_observation
                in_observation = False
            # Assign label based on current block
            elif in_thinking:
                expert_labels[i] = self.config.expert_thinking
            elif in_tool_call:
                expert_labels[i] = self.config.expert_tool_call
            elif in_observation:
                expert_labels[i] = self.config.expert_observation
            # Otherwise stays as expert_conversation (0)
        
        return expert_labels
    
    def process_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Process a dataset to add expert labels.
        
        Args:
            dataset: List of samples with 'input_ids' or 'token_ids' keys
            
        Returns:
            List of samples with added 'expert_ids' key
        """
        processed = []
        
        for sample in dataset:
            # Get token IDs from sample
            token_ids = sample.get('token_ids', sample.get('input_ids', []))
            
            if not token_ids:
                continue
            
            # Assign expert labels
            expert_labels = self.assign_expert_labels(token_ids)
            
            # Create new sample with expert labels
            new_sample = sample.copy()
            new_sample['expert_ids'] = expert_labels
            processed.append(new_sample)
        
        logger.info(f"Processed {len(processed)} samples with expert labels")
        return processed
    
    def get_expert_statistics(self, dataset: List[Dict]) -> Dict[int, Dict]:
        """Get statistics about expert label distribution.
        
        Args:
            dataset: List of samples with 'expert_ids' key
            
        Returns:
            Dictionary with statistics per expert
        """
        expert_counts = {i: 0 for i in range(self.config.num_experts)}
        total_tokens = 0
        
        for sample in dataset:
            expert_ids = sample.get('expert_ids', [])
            for expert_id in expert_ids:
                expert_counts[expert_id] = expert_counts.get(expert_id, 0) + 1
                total_tokens += 1
        
        stats = {}
        for expert_id, count in expert_counts.items():
            percentage = (count / total_tokens * 100) if total_tokens > 0 else 0
            stats[expert_id] = {
                'count': count,
                'percentage': percentage,
                'name': self._get_expert_name(expert_id)
            }
        
        return stats
    
    def _get_expert_name(self, expert_id: int) -> str:
        """Get human-readable name for an expert."""
        names = {
            0: 'conversation',
            1: 'thinking',
            2: 'tool_call',
            3: 'observation'
        }
        return names.get(expert_id, f'expert_{expert_id}')


def assign_expert_labels_to_dataset(dataset: List[Dict], tokenizer) -> List[Dict]:
    """Convenience function to assign expert labels to a dataset.
    
    Args:
        dataset: List of samples with token IDs
        tokenizer: Tokenizer instance
        
    Returns:
        List of samples with expert_ids added
    """
    processor = MoEDataProcessor(tokenizer)
    return processor.process_dataset(dataset)
