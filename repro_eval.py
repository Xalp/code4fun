import re
import math
from collections import Counter

def check_task(name, code, test_cases):
    print(f"--- Checking {name} ---")
    print(f"Code:\n{code}")
    print("Tests:")
    for t in test_cases:
        print(f"  {t}")
    
    # Create a fresh scope for execution
    scope = {}
    try:
        exec(code, scope)
    except Exception as e:
        print(f"EXECUTION ERROR: {e}")
        return

    # Run assertions
    passed = 0
    for t in test_cases:
        try:
            exec(t, scope)
            passed += 1
            print(f"  [PASS] {t}")
        except AssertionError:
            print(f"  [FAIL] Assertion Failed: {t}")
        except Exception as e:
            print(f"  [ERROR] {e}: {t}")
    
    print(f"Result: {passed}/{len(test_cases)} passed")
    print("\n")

# Data from outputs.jsonl

# Task 6: sort_matrix (Seemed correct but failed)
task6_code = """def sort_matrix(matrix):
    \"\"\"
    Sorts a given matrix in ascending order according to the sum of its rows.
    \"\"\"
    return sorted(matrix, key=lambda row: sum(row))"""
task6_tests = [
    "assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]", 
    "assert sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])==[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]", 
    "assert sort_matrix([[5,8,9],[6,4,3],[2,1,4]])==[[2, 1, 4], [6, 4, 3], [5, 8, 9]]"
]

# Task 3: remove (Uses regex, seemed correct but failed)
task3_code = """import re
def remove(strings: list) -> list:
    \"\"\"
    Removes all digits from a list of strings.
    \"\"\"
    return [re.sub('\\\\d', '', s) for s in strings]"""
task3_tests = [
    "assert remove(['4words', '3letters', '4digits']) == ['words', 'letters', 'digits']", 
    "assert remove(['28Jan','12Jan','11Jan']) == ['Jan','Jan','Jan']", 
    "assert remove(['wonder1','wonder2','wonder3']) == ['wonder','wonder','wonder']"
]

# Task 2: test_duplicate (Seemed correct but failed)
task2_code = """def test_duplicate(arr):
    \"\"\"
    Finds whether a given array of integers contains any duplicate element.
    \"\"\"
    seen = set()
    for num in arr:
        if num in seen:
            return True
        seen.add(num)
    return False"""
task2_tests = [
    "assert test_duplicate(([1,2,3,4,5]))==False", 
    "assert test_duplicate(([1,2,3,4, 4]))==True", 
    "assert test_duplicate([1,1,2,2,3,3,4,4,5])==True"
]

if __name__ == "__main__":
    check_task("sort_matrix", task6_code, task6_tests)
    check_task("remove", task3_code, task3_tests)
    check_task("test_duplicate", task2_code, task2_tests)
