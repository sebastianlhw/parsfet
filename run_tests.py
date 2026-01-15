
import sys
import os

# Add the src directory to the Python path to allow for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from tests.test_pre_compiled_regex import test_parse_time_unit, test_parse_cap_unit, test_liberty_parser_parse_cap_unit

def run_tests():
    """Runs all the tests."""
    try:
        test_parse_time_unit()
        print("test_parse_time_unit: PASSED")
        test_parse_cap_unit()
        print("test_parse_cap_unit: PASSED")
        test_liberty_parser_parse_cap_unit()
        print("test_liberty_parser_parse_cap_unit: PASSED")
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    run_tests()
