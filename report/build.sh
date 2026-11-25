#!/usr/bin/env bash
# Builds LOB_ML_Project_Report.pdf from report.tex (run twice for TOC / references).
set -euo pipefail
cd "$(dirname "$0")"
for i in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error -jobname=LOB_ML_Project_Report report.tex > build.log
done
rm -f LOB_ML_Project_Report.{aux,toc,lof,lot,lol,out}
echo "Built $(pwd)/LOB_ML_Project_Report.pdf"
