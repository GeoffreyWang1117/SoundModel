# Paper: Speech Science Beats Deep Learning

ACL/ICML 2026 submission draft

## Structure

- `main.tex`: Main paper file
- `sections/`: Individual sections
  - `abstract.tex`: Abstract (draft complete)
  - `introduction.tex`: Introduction (outline done)
  - `related_work.tex`: Related work (TODO: add citations)
  - `methodology.tex`: Method description (outline done)
  - `experimental_setup.tex`: Setup details (complete)
  - `results.tex`: Main results (outline done)
  - `analysis.tex`: Feature analysis (TODO: run experiments)
  - `discussion.tex`: Discussion (draft complete)
  - `conclusion.tex`: Conclusion (draft complete)
  - `appendix.tex`: Appendix (outline done)
- `references.bib`: Bibliography (TODO: fill)

## TODO

### Immediate (for paper completion)
1. **Analysis section experiments**:
   - [ ] Feature importance analysis (ablation studies)
   - [ ] Generalization tests (real speech)
   - [ ] Interpretability analysis

2. **Content**:
   - [ ] Fill Related Work with proper citations
   - [ ] Add figures (architecture, results plots)
   - [ ] Complete references.bib

3. **Formatting**:
   - [ ] Follow ACL/ICML style guide
   - [ ] Add proper citations throughout
   - [ ] Create figures and tables

### Later
- Review and polish writing
- Proofread
- Get feedback
- Submit to arXiv
- Submit to conference

## Building

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Note: Currently has TODOs and incomplete sections. Use as structural template.
