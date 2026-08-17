"""Quality validation for agentic data samples."""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentQualityResult:
    """Result of agentic data quality validation."""
    valid: bool
    score: float
    issues: List[str] = field(default_factory=list)


def validate_agent_sample(sample: Dict[str, Any], known_tools: Optional[List[str]] = None) -> AgentQualityResult:
    """Validate an agentic data sample.

    Checks:
    - tool_call uses valid notation: tool_name(args)
    - tool_name exists in registry
    - arguments match tool schema
    - observation is not empty
    - answer is not empty
    """
    issues = []
    score = 1.0

    thinking = sample.get('thinking', '')
    has_tool_call = sample.get('has_tool_call', False)
    tool_name = sample.get('tool_name', '')
    answer = sample.get('answer', '')
    input_text = sample.get('input_ids', '')

    if not answer:
        issues.append('empty_answer')
        score -= 0.5

    if not has_tool_call:
        # Normal sample — just check it has thinking
        if thinking:
            return AgentQualityResult(valid=True, score=score, issues=issues)
        issues.append('no_thinking')
        score -= 0.2
        return AgentQualityResult(valid=score > 0.3, score=max(0.0, score), issues=issues)

    # Validate tool_call notation (Formato 2: tool_name(args))
    tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', input_text, re.DOTALL)
    if not tool_call_match:
        issues.append('missing_tool_call_tags')
        score -= 0.5
        return AgentQualityResult(valid=False, score=max(0.0, score), issues=issues)

    tool_call_str = tool_call_match.group(1).strip()
    tool_call_parse = re.match(r'^([A-Za-z_]\w*)\s*\((.*)\)$', tool_call_str, re.DOTALL)
    if not tool_call_parse:
        issues.append('invalid_tool_call_format')
        score -= 0.5
        return AgentQualityResult(valid=False, score=max(0.0, score), issues=issues)

    name = tool_call_parse.group(1)
    args_str = tool_call_parse.group(2).strip()

    # Validate tool name
    if not name:
        issues.append('missing_tool_name')
        score -= 0.3
    elif known_tools and name not in known_tools:
        issues.append(f'unknown_tool: {name}')
        score -= 0.2

    # Validate arguments present (may be empty for parameterless tools)
    if name and name not in ('current_date', 'get_platform') and not args_str:
        issues.append('missing_arguments')
        score -= 0.2

    # Validate observation exists (prefix <|tool_result|>, runs until <|end|>/<|assistant|>)
    obs_match = re.search(r'<\|tool_result\|>(.*?)(?:<\|end\|>|<\|assistant\|>)', input_text, re.DOTALL)
    if not obs_match:
        issues.append('missing_observation')
        score -= 0.3
    elif not obs_match.group(1).strip():
        issues.append('empty_observation')
        score -= 0.2

    # Check balance: thinking should mention the tool
    if thinking and name and name not in thinking.lower() and tool_name not in thinking:
        issues.append('thinking_mismatches_tool')
        score -= 0.1

    valid = score >= 0.5 and 'invalid_tool_call_format' not in issues and 'missing_tool_call_tags' not in issues
    return AgentQualityResult(valid=valid, score=max(0.0, score), issues=issues)


def filter_low_quality_agent(samples: List[Dict[str, Any]], min_score: float = 0.5,
                              known_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Filter out low-quality agentic samples."""
    filtered = []
    for sample in samples:
        result = validate_agent_sample(sample, known_tools)
        if result.valid and result.score >= min_score:
            filtered.append(sample)
        else:
            logger.debug(f"Filtered agent sample: score={result.score}, issues={result.issues}")
    return filtered


def agent_data_report(samples: List[Dict[str, Any]], known_tools: Optional[List[str]] = None) -> Dict[str, Any]:
    """Generate a report on agentic data quality."""
    total = len(samples)
    tool_call_count = sum(1 for s in samples if s.get('has_tool_call', False))
    normal_count = total - tool_call_count

    tool_distribution = {}
    for s in samples:
        tn = s.get('tool_name', '')
        if tn:
            tool_distribution[tn] = tool_distribution.get(tn, 0) + 1

    scores = []
    for s in samples:
        r = validate_agent_sample(s, known_tools)
        scores.append(r.score)

    return {
        'total_samples': total,
        'tool_call_samples': tool_call_count,
        'normal_samples': normal_count,
        'tool_ratio': tool_call_count / total if total > 0 else 0.0,
        'tool_distribution': tool_distribution,
        'avg_quality_score': sum(scores) / len(scores) if scores else 0.0,
    }
