#!/usr/bin/env bash
# Build an OpenReview-ready zip: sources, style files, tables and the figures
# the main document actually \includegraphics, with build artefacts stripped and
# the paper/figures symlink resolved to real files.

set -e
cd "$(dirname "$0")/.."

OUT="persona_vectors_icml2026_submission.zip"
STAGE="$(mktemp -d)"
ROOT="$STAGE/persona_vectors_icml2026"
mkdir -p "$ROOT/figures" "$ROOT/tables"

cp paper/persona_vectors_icml2026.tex "$ROOT/"
cp paper/persona_vectors_icml2026.pdf "$ROOT/" 2>/dev/null || true
cp paper/refs.bib "$ROOT/"
cp paper/icml2026.sty paper/icml2026.bst paper/algorithm.sty \
   paper/algorithmic.sty paper/fancyhdr.sty "$ROOT/"
cp paper/tables/*.tex "$ROOT/tables/"

FIGURES=(
    fig1_layer_search.pdf
    fig2_magnitude.pdf
    fig_alpha_search_Qwen3-8B_LaMP-2_template.pdf
    fig_case_LaMP-2_user003_fact.pdf
    fig_operators_comparison.pdf
    fig_positive_control_Qwen3-8B_LaMP-7.pdf
    fig_variance_analysis_lamp2.pdf
    fig_variance_analysis_lamp7.pdf
)
for fig in "${FIGURES[@]}"; do
    cp "paper/figures/$fig" "$ROOT/figures/"
done

cat > "$ROOT/README.md" <<'EOF'
# Per-User Persona Vectors — ICML 2026 Workshop submission

## Files
- `persona_vectors_icml2026.tex` / `.pdf` — main document (4-page body + appendix)
- `refs.bib` — references
- `tables/` — three LaTeX tables \input from the main document
- `figures/` — the PDF figures referenced from the main document
- ICML style files: `icml2026.sty`, `icml2026.bst`, `algorithm.sty`,
  `algorithmic.sty`, `fancyhdr.sty`

## Build
```
pdflatex persona_vectors_icml2026
bibtex   persona_vectors_icml2026
pdflatex persona_vectors_icml2026
pdflatex persona_vectors_icml2026
```

For a preprint with author names, change `\usepackage{icml2026}` to
`\usepackage[preprint]{icml2026}`.
EOF

(cd "$STAGE" && zip -r -q "$OLDPWD/$OUT" persona_vectors_icml2026)
rm -rf "$STAGE"

echo "Wrote $OUT"
unzip -l "$OUT" | tail -20
