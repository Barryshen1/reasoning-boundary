import fire
from utils.request_tool import RequestOutput
from utils.tools import get_combined_granularity

def extract_incorrect_predictions(data_split="CoT", result_path=None, K=0.106, K2=0.425, mode="nl", max_examples=10):
    """
    Extract and print examples of incorrect predictions from the specified model results.
    
    Args:
        data_split: The model/method to analyze
        result_path: Path to the results file
        K: CFRB threshold
        K2: CIRB threshold
        mode: Evaluation mode ("nl", "tool", or "pot")
        max_examples: Maximum number of examples to print per category
    """
    # Use the parameters from PARAM_DICT if not directly provided
    from evaluate import PARAM_DICT
    if result_path is None and data_split in PARAM_DICT:
        K = PARAM_DICT[data_split]["K"]
        K2 = PARAM_DICT[data_split]["K2"]
        mode = PARAM_DICT[data_split]["mode"]
        result_path = PARAM_DICT[data_split]["result_path"]
    
    response_list = RequestOutput(result_path)
    
    # Categorize incorrect predictions
    incorrect_examples = {
        "CFRB (should be easy)": [],
        "PFRB (moderately difficult)": [],
        "CIRB (very difficult)": []
    }
    
    for idx in range(len(response_list)):
        origin_data = response_list.get_origin_input(idx)
        granularity = get_combined_granularity(origin_data)
        
        # Check if prediction is incorrect
        if not response_list.judge_correct(idx, mode=mode):
            # Categorize based on reasoning boundary
            if granularity <= K:
                category = "CFRB (should be easy)"
            elif granularity > K and granularity <= K2:
                category = "PFRB (moderately difficult)"
            else:
                category = "CIRB (very difficult)"
            
            # Add to the corresponding category
            if len(incorrect_examples[category]) < max_examples:
                incorrect_examples[category].append({
                    "index": idx,
                    "question": origin_data["question"],
                    "expected_answer": origin_data["answer"].split("#### ")[-1].strip(),
                    "model_response": response_list.get_last_pred_text(idx),
                    "predicted_answer": response_list.get_text_answer(idx).strip(),
                    "granularity": granularity
                })
    
    # Print the examples
    for category, examples in incorrect_examples.items():
        print(f"\n\n{'='*80}")
        print(f"CATEGORY: {category} - Found {len(examples)} examples")
        print(f"{'='*80}")
        
        for i, example in enumerate(examples):
            print(f"\nEXAMPLE {i+1}:")
            print(f"Question: {example['question']}")
            print(f"Expected Answer: {example['expected_answer']}")
            print(f"Model's Answer: {example['predicted_answer']}")
            print(f"Granularity Value: {example['granularity']}")
            print(f"{'='*50}")
    
    # Summary statistics
    print("\n\nSUMMARY:")
    for category, examples in incorrect_examples.items():
        print(f"{category}: {len(examples)} examples found")

if __name__ == "__main__":
    fire.Fire(extract_incorrect_predictions)
