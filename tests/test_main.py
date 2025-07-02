import sys
import logging
import argparse
import glob
import importlib
from pathlib import Path

from utils import run_test

import chatlearn


def _test_func(path, case_name):
    # Traverse all test_.*.py test files in target path
    test_modules = sorted([
        Path(f).stem for f in glob.glob(f"{path}/test_*.py")
        if not Path(f).stem.startswith("__") ])

    test_cases = []
    for module_name in test_modules:
        try:
            module = importlib.import_module(f"{path.replace('/', '.')}.{module_name}")
            test_case = getattr(module, "TEST_CASE")
            test_cases.extend(test_case)
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"加载失败: {module_name} ({type(e).__name__})")
    
    # init chatlearn framework once
    chatlearn.init()

    return run_test(test_cases, case_name)

def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t", "--test_case",
        type=str, required=True, help="test_case or test_case.case_name")
    return parser.parse_known_args()

def _run_unit_tests(test_dir):
    import unittest
    import os
    discover_config = {
        "start_dir": test_dir,
        "pattern": "test_*.py",
        "top_level_dir": None
    }

    test_suite = unittest.defaultTestLoader.discover(**discover_config)

    runner = unittest.TextTestRunner(
        verbosity=2,
        failfast=False,
        buffer=False,
        resultclass=None
    )

    test_result = runner.run(test_suite)

    print(f"Total unittest cases: {test_result.testsRun}")
    print(f"Failed unittest cases: {len(test_result.failures)}")
    print(f"Error unittest cases: {len(test_result.errors)}")
    if len(test_result.failures) == 0 and len(test_result.errors) == 0:
        return 0 # UT Passed
    return 1

if __name__ == "__main__":
    args, _ = _parse_args()
    test_case = args.test_case
    case_name = None
    args_split = test_case.split('.')
    if (len(args_split) == 2):
        test_case = args_split[0]
        case_name = args_split[1]

    if test_case == "unittest":
        sys.exit(_run_unit_tests("unittests"))
    sys.exit(_test_func(test_case, case_name))