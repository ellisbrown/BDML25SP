#!/bin/bash
set -e

print_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --repo-id <name>      ID of the HF repo (e.g., username/dataset-name)"
    echo "  --type <name>         Type of repo (dataset or model, default: dataset)"
    echo "  --path <directory>    Local directory to upload (default: current directory)"
    echo "  --private             Make the repository private"
    echo "  --message <text>      Commit message (default: 'Upload dataset')"
    echo "  --help                Show help"
}

# Default values
TYPE="dataset"
LOCAL_PATH="."
PRIVATE=""
COMMIT_MESSAGE="Upload dataset"

while [[ $# -gt 0 ]]; do
    case $1 in
        --repo-id)
            REPO_ID="$2"
            shift 2 ;;
        --type)
            TYPE="$2"
            shift 2 ;;
        --path)
            LOCAL_PATH="$2"
            shift 2 ;;
        --private)
            PRIVATE="--private"
            shift ;;
        --message)
            COMMIT_MESSAGE="$2"
            shift 2 ;;
        --help)
            print_help
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1 ;;
    esac
done

# Check for required parameters
[ -z "$REPO_ID" ] && { echo "Error: Repository ID required (--repo-id)"; exit 1; }

# Validate repo type
if [[ "$TYPE" != "dataset" && "$TYPE" != "model" ]]; then
    echo "Error: Type must be 'dataset' or 'model'"
    exit 1
fi

# Check if local path exists
if [ ! -d "$LOCAL_PATH" ]; then
    echo "Error: Directory $LOCAL_PATH does not exist"
    exit 1
fi

echo "Uploading to $REPO_ID... (type: $TYPE)"
echo "Source directory: $LOCAL_PATH"
echo "Privacy setting: ${PRIVATE:-public}"

# # Create repository if it doesn't exist
# if ! huggingface-cli repo info $REPO_ID --type=$TYPE &>/dev/null; then
#     echo "Repository doesn't exist, creating it..."
#     huggingface-cli repo create $REPO_ID --type=$TYPE $PRIVATE
# fi

# Enable fast file uploads (similar to fast downloads)
export HF_HUB_ENABLE_HF_TRANSFER=1

# Upload the files
huggingface-cli upload $REPO_ID $LOCAL_PATH --repo-type=$TYPE --commit-message="$COMMIT_MESSAGE"

echo "Upload complete!"
