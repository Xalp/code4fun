import re
from typing import Union

import evaluate as hf_evaluate
from typing import List
from lm_eval.tasks.humaneval.sanitize_utils import sanitize


try:
    pass_at_k = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(
    references: Union[str, list[str]], predictions: Union[str, list[list[str]]]
) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


def extract_code_blocks(text: str) -> str:
    # Pattern to match ```...``` blocks
    pattern = r"```(?:\w+)?\n?(.*?)\n?```"
    # (+ ```) as we add the opening "```python" to the gen_prefix
    matches = re.findall(pattern, text, re.DOTALL)
    # if no matches, try to match ```...``` blocks (after removing the language)
    if not matches:
        text_without_lang = re.sub(r"```python", "```", text)
        matches = re.findall(pattern, text_without_lang, re.DOTALL)
    if not matches:
        return ""
    else:
        return matches[0]


class LLaDAExtractCodeBlocks:
    def apply(self, resps: List[str], docs: List[dict]) -> List[str]:
        def _extract_one(text: str) -> str:
            if isinstance(text, list):
                text = text[0] if len(text) > 0 else ""
            pattern = r"```(?:\w+)?\n?(.*?)\n?```"
            matches = re.findall(pattern, text[0], re.DOTALL)
            if not matches:
                text_without_lang = re.sub(r"```python", "```", text)
                matches = re.findall(pattern, text_without_lang, re.DOTALL)
            return matches[0].strip() if matches else text.strip()

        return [_extract_one(r) for r in resps]



def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_code_blocks(r) for r in resp] for resp in resps]

def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    results = []
    for resp, doc in zip(resps, docs):
        sanitized_resps = []
        for r in resp:
            # 1. Extract code block
            # If model follows instruction, it might output "arg): body" or "```python\narg): body```"
            # We strip markdown first.
            if "```python" in r:
                code = r.split("```python")[-1]
                if "```" in code:
                    code = code.split("```")[0]
            elif "```" in r:
                code = r.split("```")[1] 
                if "```" in code:
                    code = code.split("```")[0]
            else:
                code = r
            
            code = code.strip()

            # 2. Extract expected function name from tests
            target_name = None
            if "test_list" in doc and len(doc["test_list"]) > 0:
                match = re.search(r"assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", doc["test_list"][0])
                if match:
                    target_name = match.group(1)

            # 3. Reconstruct if missing def (continuation)
            if target_name and not code.startswith("def "):
                 # Check if it looks like argument list continuation or just body
                 # If prompt ended in 'def name(', model starts with 'arg1, arg2):'
                 # We simply prepend the def start.
                 # However, if code starts with ')', it implies 0 args?
                 # Safest is to just prepend.
                 code = f"def {target_name}({code}"
            
            # 4. Sanitize (now we have full code)
            code = sanitize(code, entrypoint=None)

            # 5. Alias check (if model wrote its own def with wrong name)
            generated_name = None
            try:
                tree = ast.parse(code)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        generated_name = node.name
                        break 
            except:
                pass

            if target_name and generated_name and target_name != generated_name:
                code += f"\n{target_name} = {generated_name}\n"
            
            sanitized_resps.append(code)
        results.append(sanitized_resps)
    return results

def process_docs_instruct(dataset):
    def _helper(doc):
        # 1. Prompt (User)
        instruction = "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function."
        text = doc.get('text', "")
        test_list = doc.get('test_list', [])
        
        tests_str = ""
        if len(test_list) > 0:
            tests_str = "Your code should pass these tests:\n\n" + "\n".join(test_list[:3])
        
        doc["instruct_prompt"] = f"{instruction}\n{text}\n{tests_str}"

        # 2. Prefix (Assistant)
        func_name = "solution"
        if len(test_list) > 0:
            match = re.search(r"assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", test_list[0])
            if match:
                func_name = match.group(1)
        
        doc["func_prefix"] = f" Here is the completed function:\n```python\ndef {func_name}("
        return doc

    return dataset.map(_helper)


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
