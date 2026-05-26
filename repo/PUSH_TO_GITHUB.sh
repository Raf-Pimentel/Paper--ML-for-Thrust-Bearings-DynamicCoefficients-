#!/usr/bin/env bash
# =============================================================================
# PUSH_TO_GITHUB.sh
# =============================================================================
# Applies the new research-grade project structure on top of your EXISTING
# repository, preserving all previous commits and files.
#
# How it works:
#   1. Clones your current remote repo into a temp folder
#   2. Copies every new file from this folder into the clone
#   3. Stages, commits, and pushes — no --force, history fully preserved
#
# Prerequisites: git and your GitHub credentials (or SSH key) configured.
#
# Run from the folder that CONTAINS this script:
#   bash PUSH_TO_GITHUB.sh
# =============================================================================
set -e

REMOTE_URL="https://github.com/Raf-Pimentel/Paper--ML-for-Thrust-Bearings-DynamicCoefficients-.git"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLONE_DIR="$(mktemp -d)/repo_clone"

echo ""
echo "=========================================="
echo " MECSOL 2026 — GitHub structure update"
echo "=========================================="
echo ""

# ── Step 1: clone the current remote ─────────────────────────────────────────
echo "[1/4] Cloning existing repository..."
git clone "$REMOTE_URL" "$CLONE_DIR"
echo "      Cloned to $CLONE_DIR"

# ── Step 2: copy new files into the clone ────────────────────────────────────
echo ""
echo "[2/4] Copying new files into the clone..."

# Copy everything from this folder except the script itself and hidden dirs
rsync -av --exclude='.git' \
          --exclude='PUSH_TO_GITHUB.sh' \
          --exclude='__pycache__' \
          --exclude='.pytest_cache' \
          "$SCRIPT_DIR/" "$CLONE_DIR/"

echo "      Files copied."

# ── Step 3: stage and commit ──────────────────────────────────────────────────
echo ""
echo "[3/4] Staging and committing..."
cd "$CLONE_DIR"

git add -A

# Only commit if there is something to commit
if git diff --cached --quiet; then
    echo "      Nothing new to commit — repository already up to date."
else
    git commit -m "refactor: international-grade research project structure

New layout applied on top of existing files:
  src/solver/reynolds_solver.py    — documented 2-D FDM/SOR solver (class API)
  src/utils/analytical.py         — SymPy-verified 1-D formulas (corrected scale)
  src/ml/generate_dataset.py      — 4,500-pt sweep with tqdm + checkpoint resume
  src/ml/train_surrogate.py       — RF / XGBoost / ANN training + export
  src/visualization/              — publication-ready figure scripts (Figs 5–8)
  tests/test_solver.py            — 13 pytest unit tests (all passing)
  docs/equations.md               — full mathematical derivation reference
  requirements.txt                — pinned dependencies
  environment.yml                 — conda environment spec
  README.md                       — international-standard docs with results table,
                                    quick-start guide, physics background, BibTeX

Existing files (src/2S_2025/, Mancais_Dashboard/, figures/, docs/)
preserved unchanged.

MECSOL 2026 — LAMAR / FEM-UNICAMP"
    echo "      Commit created."
fi

# ── Step 4: push ──────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Pushing to origin/main..."
git push origin main
echo ""
echo "=========================================="
echo " Done!  $REMOTE_URL"
echo "=========================================="
