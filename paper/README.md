# Public preprint

- [PDF](PDE_OBS_preprint.pdf)
- [LaTeX source ZIP](PDE_OBS_arxiv_source.zip)

The named public version preserves the paper's scientific text, numerical results
and figures. Ruichen Xu and Yuefan Deng are the corresponding authors. It uses
the ICLR template in named-author mode with a **Preprint** header, not a claim of
conference acceptance. The additional author block yields 10 main-text pages
and 60 total pages; the separately maintained anonymous submission is unchanged.

No arXiv identifier has been assigned by this release task, and the manuscript
has not been submitted to arXiv by it. The ZIP was compiled locally from an
isolated source directory. After extracting, compile its root `main.tex` with
pdfLaTeX and BibTeX:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

The bundled `main.bbl`, figures and local styles are included. An arXiv upload
still requires author approval, a manuscript license choice and verification of
arXiv's generated preview. The repository's MIT code license is not an arXiv
manuscript-license selection.
