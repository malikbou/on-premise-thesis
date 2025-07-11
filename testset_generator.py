import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass
class SingleTurnSample:
    user_input: str
    retrieved_contexts: Optional[List[str]] = None
    reference_contexts: List[str] = None
    response: Optional[str] = None
    multi_responses: Optional[Dict[str, str]] = None
    reference: str = None
    rubrics: Optional[Dict[str, Any]] = None

def convert_samples_to_json(samples: List[str], output_path: str = "data/testset.json"):
    """
    Convert a list of SingleTurnSample string representations to a JSON file

    Args:
        samples: List of strings representing SingleTurnSample objects
        output_path: Path to save the JSON output
    """
    # Parse the sample strings and extract user_input and reference
    output_data = []

    for sample_str in samples:
        # Extract sample from the string representation
        if "eval_sample=SingleTurnSample" in sample_str:
            # Parse the string to extract user_input and reference
            user_input = extract_field(sample_str, "user_input")
            reference = extract_field(sample_str, "reference")

            if user_input and reference:
                output_data.append({
                    "user_input": user_input,
                    "reference": reference
                })

    # Save the testset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\nSuccessfully generated and saved {len(output_data)} Q&A pairs to {output_path}")
    return output_data

def extract_field(sample_str: str, field_name: str) -> str:
    """Extract a field value from the SingleTurnSample string representation"""
    start_marker = f"{field_name}='"
    if start_marker not in sample_str:
        start_marker = f"{field_name}=\""

    if start_marker not in sample_str:
        return None

    start_idx = sample_str.find(start_marker) + len(start_marker)

    # Find the closing quote
    if sample_str[start_idx-1] == "'":
        end_idx = sample_str.find("'", start_idx)
    else:
        end_idx = sample_str.find('"', start_idx)

    if end_idx == -1:
        return None

    return sample_str[start_idx:end_idx]

if __name__ == "__main__":
    # Example usage
    samples = [
        """eval_sample=SingleTurnSample(user_input='What are the key responsibilities of an Academic Assessment Coordinator in ensuring accurate grading of student work?', retrieved_contexts=None, reference_contexts=[], response=None, multi_responses=None, reference='The key responsibilities of an Academic Assessment Coordinator in ensuring accurate grading of student work include ensuring marking guidelines are followed for individual items of assessment and providing criteria for grading student work.', rubrics=None) synthesizer_name='multi_hop_abstract_query_synthesizer'""",

        """eval_sample=SingleTurnSample(user_input='What are the key criteria outlined by the departmental guidelines for assessing student work?', retrieved_contexts=None, reference_contexts=[], response=None, multi_responses=None, reference='The key criteria outlined by the departmental guidelines for assessing student work include factors such as originality, depth of analysis, adherence to academic standards, and overall coherence of arguments presented in the work.', rubrics=None) synthesizer_name='multi_hop_abstract_query_synthesizer'""",

        """eval_sample=SingleTurnSample(user_input='There are no themes available for combination in this context.', retrieved_contexts=None, reference_contexts=[], response=None, multi_responses=None, reference='No concepts available for combination.', rubrics=None) synthesizer_name='multi_hop_abstract_query_synthesizer'"""
    ]

    convert_samples_to_json(samples)
