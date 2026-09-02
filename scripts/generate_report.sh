#!/bin/bash
# Generate HTML training report from CSV metrics
# Usage: ./generate_report.sh [csv_path] [--draft]
# Example: ./generate_report.sh
# Example: ./generate_report.sh checkpoints/chat_model_metrics.csv
# Example: ./generate_report.sh checkpoints/draft_model_metrics.csv --draft

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CSV_PATH="${1:-checkpoints/chat_model_metrics.csv}"
IS_DRAFT="${2:-}"

if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: CSV file not found: $CSV_PATH"
    echo ""
    echo "Usage: $0 [csv_path] [--draft]"
    echo "Example: $0 checkpoints/chat_model_metrics.csv"
    exit 1
fi

echo "Generating report from: $CSV_PATH"
python "$SCRIPT_DIR/generate_report.py" "$CSV_PATH" $IS_DRAFT

if [ $? -eq 0 ]; then
    HTML_PATH="${CSV_PATH%.csv}_report.html"
    echo ""
    echo "Done! Opening in browser..."
    if command -v xdg-open &> /dev/null; then
        xdg-open "$HTML_PATH"
    elif command -v open &> /dev/null; then
        open "$HTML_PATH"
    else
        echo "Open manually: file://$(pwd)/$HTML_PATH"
    fi
else
    echo "ERROR: Failed to generate report"
    exit 1
fi
