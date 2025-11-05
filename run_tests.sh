#!/bin/bash
# Test runner script for Tubify

set -e

echo "Tubify Test Suite"
echo "================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest not found. Install with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Default: run all tests
TEST_PATH="${1:-}"

if [ -z "$TEST_PATH" ]; then
    echo "Running all tests..."
    pytest
else
    echo "Running tests in: $TEST_PATH"
    pytest "$TEST_PATH"
fi

echo ""
echo "Tests complete!"
