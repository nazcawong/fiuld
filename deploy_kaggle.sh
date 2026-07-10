#!/bin/bash
set -e

echo "🚀 Fiuld Kaggle Deployment Script"
echo "==================================="

# Paths
EXECUTABLE_NAME="fiuld"
DATASET_NAME="fiuld-binary-dataset"
TEST_JSON="/kaggle/input/arc-prize-2026/test.json"
SUBMISSION_OUTPUT="/kaggle/working/submission.json"

# Check executable exists
if [ ! -f "$EXECUTABLE_NAME" ]; then
    echo "❌ Error: $EXECUTABLE_NAME not found!"
    exit 1
fi

# Make executable
chmod +x "$EXECUTABLE_NAME"
echo "✅ Executable ready: $(stat -f%z "$EXECUTABLE_NAME" 2>/dev/null || stat -c%s "$EXECUTABLE_NAME") bytes"

# Check test.json
if [ ! -f "$TEST_JSON" ]; then
    echo "❌ Error: $TEST_JSON not found!"
    echo "   Available inputs:"
    ls -la /kaggle/input/ 2>/dev/null || echo "   (no /kaggle/input directory)"
    exit 1
fi

echo "📂 Test JSON: $TEST_JSON"
echo "📤 Output: $SUBMISSION_OUTPUT"
echo ""

# Run engine
echo "⚔️ Launching Fiuld engine..."
START_TIME=$(date +%s)

./"$EXECUTABLE_NAME" "$TEST_JSON" "$SUBMISSION_OUTPUT"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "🏁 Engine completed in ${MINUTES}m ${SECONDS}s"

# Validate output
if [ -f "$SUBMISSION_OUTPUT" ]; then
    FILE_SIZE=$(stat -f%z "$SUBMISSION_OUTPUT" 2>/dev/null || stat -c%s "$SUBMISSION_OUTPUT")
    echo "✅ Submission file: $(echo "scale=2; $FILE_SIZE / 1024" | bc) KB"
    echo "   Predictions: $(python3 -c "import json; d=json.load(open('$SUBMISSION_OUTPUT')); print(len(d))" 2>/dev/null || echo "unknown")"
    echo ""
    echo "🎯 Ready for upload!"
else
    echo "❌ Error: submission.json not generated!"
    exit 1
fi
