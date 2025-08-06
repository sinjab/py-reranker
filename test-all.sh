#!/bin/bash

# Test all rerankers with all test files
echo "🔍 Testing all rerankers with all test data files..."
echo "=================================================="

for file in tests/data/*.json; do
    if [ -f "$file" ]; then
        echo ""
        echo "📄 Testing with: $(basename "$file")"
        echo "----------------------------------------"
        uv run python main.py --test-file "$file"
    fi
done

echo ""
echo "✅ All tests completed!"
