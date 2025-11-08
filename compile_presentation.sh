#!/bin/bash
# Script to compile Beamer presentation to PDF

echo "Compiling Beamer presentation..."

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found. Please install LaTeX (texlive-full)."
    echo "On Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "On macOS: brew install --cask mactex"
    exit 1
fi

# Compile (run twice for references)
pdflatex -interaction=nonstopmode presentation_slides.tex
pdflatex -interaction=nonstopmode presentation_slides.tex

# Clean up auxiliary files
rm -f presentation_slides.aux presentation_slides.log presentation_slides.nav \
      presentation_slides.out presentation_slides.snm presentation_slides.toc

echo "✓ Compilation complete: presentation_slides.pdf"
