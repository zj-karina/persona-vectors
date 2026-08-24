#!/usr/bin/env bash
# One-shot bootstrap: init git, make the first commit, create the GitHub repo.
# After this, use git and gh normally.
#
#   bash scripts/init_repo.sh
#   VISIBILITY=private REPO_NAME=my-name bash scripts/init_repo.sh

set -e
cd "$(dirname "$0")/.."

REPO_NAME="${REPO_NAME:-$(basename "$(pwd)")}"
VISIBILITY="${VISIBILITY:-public}"

echo "=== Init git repo ==="
if [ ! -d .git ]; then
    git init -b main
fi

# Fall back to the gh account identity when git has none configured.
if [ -z "$(git config user.name 2>/dev/null)" ]; then
    NAME="${GIT_NAME:-$(gh api user -q .name 2>/dev/null)}"
    [ -z "$NAME" ] && NAME=$(gh api user -q .login 2>/dev/null)
    git config user.name "$NAME"
fi
if [ -z "$(git config user.email 2>/dev/null)" ]; then
    EMAIL="${GIT_EMAIL:-$(gh api user -q .email 2>/dev/null)}"
    if [ -z "$EMAIL" ] || [ "$EMAIL" = "null" ]; then
        EMAIL="$(gh api user -q .id)+$(gh api user -q .login)@users.noreply.github.com"
    fi
    git config user.email "$EMAIL"
fi
echo "  using: $(git config user.name) <$(git config user.email)>"

echo "=== Verify no secrets ==="
if git ls-files --others --exclude-standard --modified --cached 2>/dev/null \
    | xargs grep -l -E "hf_[a-zA-Z0-9]{30,}" 2>/dev/null; then
    echo "!! Found suspected HF token in tracked files. Aborting."
    exit 1
fi

echo "=== Stage and commit ==="
git add .
git commit -m "Initial commit: $REPO_NAME"

echo "=== Push to GitHub ==="
if ! gh auth status >/dev/null 2>&1; then
    echo "!! Not logged in to GitHub. Run:  gh auth login"
    exit 1
fi

if gh repo view "$REPO_NAME" >/dev/null 2>&1; then
    echo "Repo $REPO_NAME exists; setting remote and pushing."
    git remote add origin "git@github.com:$(gh api user -q .login)/$REPO_NAME.git" 2>/dev/null || true
    git push -u origin main
else
    gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
fi

echo ""
echo "Done. Repo: $(gh repo view --json url -q .url)"
