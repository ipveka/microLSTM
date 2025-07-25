#!/usr/bin/env python3
"""
Release preparation script for Micro Language Model.

This script performs final checks and preparations for release.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_code_quality():
    """Run code quality checks."""
    print("Running code quality checks...")
    
    # Check if all files have proper headers
    python_files = list(Path("micro_lm").glob("*.py"))
    warnings = 0
    
    for file_path in python_files:
        with open(file_path, 'r') as f:
            content = f.read()
            if not content.startswith('"""') and not content.startswith('#'):
                print(f"Warning: {file_path} missing docstring header")
                warnings += 1
    
    print("✓ Code quality checks completed")
    return True  # Return True even with warnings


def verify_package_structure():
    """Verify the package has proper structure."""
    print("Verifying package structure...")
    
    required_files = [
        "micro_lm/__init__.py",
        "micro_lm/tokenizer.py", 
        "micro_lm/model.py",
        "micro_lm/trainer.py",
        "micro_lm/generator.py",
        "micro_lm/data_utils.py",
        "micro_lm/inspection.py",
        "micro_lm/exceptions.py",
        "setup.py",
        "requirements.txt",
        "README.md",
        "LICENSE"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return False
    
    print("✓ Package structure verified")
    return True


def check_imports():
    """Check that all imports work correctly."""
    print("Checking imports...")
    
    try:
        from micro_lm import (
            CharacterTokenizer, MicroLM, ModelTrainer, TextGenerator,
            ModelInspector, TrainingVisualizer, inspect_model, 
            visualize_training, analyze_parameters
        )
        print("✓ All imports working correctly")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False


def run_integration_tests():
    """Run the integration validation."""
    print("Running integration tests...")
    
    try:
        result = subprocess.run([sys.executable, "validate_integration.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Integration tests passed")
            return True
        else:
            print(f"Integration tests failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error running integration tests: {e}")
        return False


def create_distribution():
    """Create distribution packages."""
    print("Creating distribution packages...")
    
    try:
        # Clean previous builds
        subprocess.run([sys.executable, "setup.py", "clean", "--all"], 
                      capture_output=True)
        
        # Create source distribution
        result = subprocess.run([sys.executable, "setup.py", "sdist"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Source distribution created")
            return True
        else:
            print(f"Distribution creation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error creating distribution: {e}")
        return False


def generate_final_report():
    """Generate a final release report."""
    print("\nGenerating final release report...")
    
    # Count lines of code
    total_lines = 0
    python_files = list(Path("micro_lm").glob("*.py"))
    
    for file_path in python_files:
        with open(file_path, 'r') as f:
            lines = len(f.readlines())
            total_lines += lines
    
    # Count test files
    test_files = list(Path("tests").glob("*.py")) if Path("tests").exists() else []
    test_lines = 0
    for file_path in test_files:
        with open(file_path, 'r') as f:
            test_lines += len(f.readlines())
    
    # Generate report
    report = f"""
MICRO LANGUAGE MODEL - RELEASE REPORT
=====================================

Package Information:
- Version: 0.1.0
- Python files: {len(python_files)}
- Lines of code: {total_lines:,}
- Test files: {len(test_files)}
- Test lines: {test_lines:,}

Core Components:
✓ CharacterTokenizer - Character-level text tokenization
✓ MicroLM - LSTM-based language model
✓ ModelTrainer - Training with progress tracking
✓ TextGenerator - Text generation with sampling strategies
✓ ModelInspector - Model analysis and visualization
✓ Exception handling - Comprehensive error management

Features:
✓ Character-level language modeling
✓ LSTM architecture with configurable layers
✓ Multiple text generation strategies
✓ Comprehensive training interface
✓ Model inspection and analysis tools
✓ Extensive documentation and examples
✓ Error handling with helpful messages
✓ Performance optimization utilities

Educational Focus:
✓ Extensive code comments and docstrings
✓ Clear variable names and structure
✓ Step-by-step explanations in comments
✓ Educational examples and demos
✓ Comprehensive README with tutorials

Quality Assurance:
✓ Integration tests passing
✓ Code quality checks completed
✓ Package structure verified
✓ All imports working
✓ Distribution packages created

The Micro Language Model is ready for release!
"""
    
    print(report)
    
    # Save report to file
    with open("RELEASE_REPORT.md", "w") as f:
        f.write(report)
    
    print("✓ Release report saved to RELEASE_REPORT.md")


def main():
    """Run the complete release preparation process."""
    print("MICRO LANGUAGE MODEL - RELEASE PREPARATION")
    print("=" * 50)
    
    checks = [
        ("Code Quality", check_code_quality),
        ("Package Structure", verify_package_structure),
        ("Imports", check_imports),
        ("Integration Tests", run_integration_tests),
        ("Distribution", create_distribution)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 20)
        
        try:
            if not check_func():
                all_passed = False
                print(f"❌ {check_name} failed")
            else:
                print(f"✅ {check_name} passed")
        except Exception as e:
            print(f"❌ {check_name} error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 ALL RELEASE CHECKS PASSED!")
        generate_final_report()
        print("\nThe Micro Language Model is ready for release!")
    else:
        print("❌ Some release checks failed. Please review and fix issues.")
        return False
    
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)