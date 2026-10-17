#!/bin/sh
# Push a clean snapshot of the project to the public repository (git remote "public").
#
# The course repository's history includes course-only files (session reporting
# settings, session transcripts, the course commit hook, course instructions), so it
# cannot be published as is. This builds a commit from the current main branch without
# those paths and adds it on top of the public repository's own history, which never
# contained them. No history is rewritten: each run is one fast-forward commit.
#
# Usage: scripts/publish_public.sh "message for the public commit"
set -eu

EXCLUDED_PATHS=".claude .claude-sessions .githooks CLAUDE.md"
message="${1:-Update from main}"

cd "$(git rev-parse --show-toplevel)"
git fetch --quiet public

# Build the snapshot's tree in a throwaway index so the working copy is untouched.
export GIT_INDEX_FILE="$(git rev-parse --git-dir)/public-index"
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
git read-tree main
git rm --cached -r -q --ignore-unmatch $EXCLUDED_PATHS
tree=$(git write-tree)

if git rev-parse -q --verify refs/remotes/public/main >/dev/null; then
    parent=$(git rev-parse refs/remotes/public/main)
    if [ "$(git rev-parse "$parent^{tree}")" = "$tree" ]; then
        echo "public repository is already up to date"
        exit 0
    fi
    commit=$(git commit-tree "$tree" -p "$parent" -m "$message")
else
    commit=$(git commit-tree "$tree" -m "$message")
fi

# Refuse to publish if any excluded path slipped into the snapshot.
for path in $EXCLUDED_PATHS; do
    if git ls-tree -r --name-only "$commit" | grep -q "^$path\(/\|$\)"; then
        echo "error: $path is in the snapshot; not publishing" >&2
        exit 1
    fi
done

git push public "$commit:refs/heads/main"
echo "published $(git rev-parse --short "$commit") to $(git remote get-url public)"
