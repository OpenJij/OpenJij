
import subprocess
import pytest
'''
def test_cpp_tests():
    result = subprocess.run(['./build/tests'], capture_output=True, text=True)
    assert result.returncode == 0, f"C++ tests failed:\n{result.stdout}\n{result.stderr}"

'''

import os
'''
def test_cpp_tests():
    # Define the path to the test binary
    test_binary_path = './build/tests'
    
    # Ensure the test binary is executable
    assert os.path.exists(test_binary_path), f"Test binary not found at {test_binary_path}"
    assert os.access(test_binary_path, os.X_OK), f"Test binary is not executable: {test_binary_path}"
    
    # Run the test binary and capture the output
    result = subprocess.run([test_binary_path], capture_output=True, text=True)
    
    # Print the output for debugging purposes
    print(result.stdout)
    print(result.stderr)
    
    # Assert the test run was successful
    assert result.returncode == 0, f"Tests failed with return code {result.returncode}"
if __name__ == "__main__":
    pytest.main([__file__])

'''
import subprocess
import os
import pytest
def test_cpp_tests():
    # Define the path to the test binary
    test_binary_path = './build/tests/cxxjij_test'
    
    # Ensure the test binary exists and is executable
    assert os.path.exists(test_binary_path), f"Test binary not found at {test_binary_path}"
    assert os.access(test_binary_path, os.X_OK), f"Test binary is not executable: {test_binary_path}"
    
    # Run the test binary and capture the output
    result = subprocess.run([test_binary_path], capture_output=True, text=True)
    
    # Print the output for debugging purposes
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    
    # Assert the test run was successful
    assert result.returncode == 0, f"Tests failed with return code {result.returncode}"

if __name__ == "__main__":
    pytest.main([__file__])

