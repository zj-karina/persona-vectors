#!/usr/bin/env bash
# Compile paper/persona_vectors_icml2026.tex, preferring a local pdflatex and
# falling back to the texlive Docker image.
#
# bibtex needs the .aux from the first pass, and two further passes settle the
# cross-references it produces.

set -e
cd "$(dirname "$0")/.."

TEX="persona_vectors_icml2026"
PASSES="pdflatex -interaction=nonstopmode $TEX.tex \
    && bibtex $TEX \
    && pdflatex -interaction=nonstopmode $TEX.tex \
    && pdflatex -interaction=nonstopmode $TEX.tex"

if command -v pdflatex >/dev/null 2>&1; then
    bash -c "cd paper && $PASSES"
    echo "=== Done: paper/$TEX.pdf ==="
elif command -v docker >/dev/null 2>&1; then
    echo "=== building via Docker (texlive/texlive:latest) ==="
    docker run --rm -v "$(pwd)":/work -w /work texlive/texlive:latest \
        bash -c "cd paper && $PASSES"
    echo "=== Done: paper/$TEX.pdf ==="
else
    cat <<'MSG'
!! No pdflatex and no docker found.
Options:
  1. Local TeX:    sudo apt install texlive-latex-recommended texlive-publishers texlive-science
  2. Or upload paper/ + figures/ to overleaf.com and compile there.
MSG
    exit 1
fi
