#!/usr/bin/env python3
"""
Test runner for the turbo-octo-funicular project.

This script runs all unit tests for the core files:
- dataset.py
- model.py 
- train.py

Usage:
    python run_tests.py
    
Or run individual test modules:
    python -m unittest test_dataset.py -v
    python -m unittest test_model.py -v  
    python -m unittest test_train.py -v
"""

import unittest
import sys
import os

def run_all_tests():
    """Run all unit tests and return the result."""
    # Discover and run all tests
    test_suite = unittest.TestLoader().discover('.', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running unit tests for turbo-octo-funicular...")
    print("=" * 60)
    
    success = run_all_tests()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
        sys.exit(1)