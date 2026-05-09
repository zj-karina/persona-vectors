#!/usr/bin/env bash
# Build a clean OpenReview-ready submission zip.
#
# - Resolves paper/figures symlink to real files
# - Keeps style files, .tex, .bib, .pdf, tables/, figures/
# - Strips build artefacts (.aux/.bbl/.log/...)
#
# Usage:
#   bash scripts/package_submission.sh

set -e
cd "$(dirname "$0")/.."

OUT="persona_vectors_icml2026_submission.zip"
STAGE="$(mktemp -d)"
ROOT="$STAGE/persona_vectors_icml2026"
mkdir -p "$ROOT/figures" "$ROOT/tables"

# .tex + .bib + style files + compiled PDF
cp paper/persona_vectors_icml2026.tex "$ROOT/"
cp paper/persona_vectors_icml2026.pdf "$ROOT/" 2>/dev/null || true
cp paper/refs.bib "$ROOT/"
cp paper/icml2026.sty "$ROOT/"
cp paper/icml2026.bst "$ROOT/"
cp paper/algorithm.sty "$ROOT/"
cp paper/algorithmic.sty "$ROOT/"
cp paper/fancyhdr.sty "$ROOT/"

# Tables
cp paper/tables/*.tex "$ROOT/tables/"

# Figures (resolve the symlink → copy real files)
cp paper/figures/fig_case_LaMP-2_user003_fact.pdf             "$ROOT/figures/"
cp paper/figures/fig2_magnitude.pdf                            "$ROOT/figures/"
cp paper/figures/fig_positive_control_Qwen3-8B_LaMP-7.pdf      "$ROOT/figures/"
cp paper/figures/fig1_layer_search.pdf                         "$ROOT/figures/"
cp paper/figures/fig_alpha_search_Qwen3-8B_LaMP-2_template.pdf "$ROOT/figures/"
cp paper/figures/fig_operators_comparison.pdf                  "$ROOT/figures/"
cp paper/figures/fig_variance_analysis_lamp2.pdf               "$ROOT/figures/"
cp paper/figures/fig_variance_analysis_lamp7.pdf               "$ROOT/figures/"

# README inside the zip
cat > "$ROOT/README.md" <<'EOF'
# Per-User Persona Vectors — ICML 2026 Workshop submission

## Files
- `persona_vectors_icml2026.tex` / `.pdf` — main document (4-page body + appendix)
- `refs.bib` — references
- `tables/` — three LaTeX tables \input from the main document
- `figures/` — four PDF figures referenced from the main document
- ICML style files: `icml2026.sty`, `icml2026.bst`, `algorithm.sty`,
  `algorithmic.sty`, `fancyhdr.sty`

## Build
```
pdflatex persona_vectors_icml2026
bibtex   persona_vectors_icml2026
pdflatex persona_vectors_icml2026
pdflatex persona_vectors_icml2026
```

For preprint with author names, change `\usepackage{icml2026}` to
`\usepackage[preprint]{icml2026}`.
EOF

cd "$STAGE" && zip -r -q "$OLDPWD/$OUT" persona_vectors_icml2026 && cd "$OLDPWD"
rm -rf "$STAGE"

echo "Wrote $OUT"
unzip -l "$OUT" | tail -20
