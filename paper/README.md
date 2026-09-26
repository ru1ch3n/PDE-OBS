# Public preprint

- [PDF](PDE_OBS_preprint.pdf)
- [LaTeX source ZIP](PDE_OBS_arxiv_source.zip)
- [Dataset](https://huggingface.co/datasets/ru1ch3n/PDE-OBS)
- [Model checkpoints](https://huggingface.co/ru1ch3n/PDE-OBS)

The named public version preserves the paper's scientific text, numerical results
and figures. Ruichen Xu and Yuefan Deng are the corresponding authors. It uses
the single-column [arxiv-style](https://github.com/kourgeorge/arxiv-style)
preprint template with a **Preprint** header, not a claim of conference
acceptance. The main paper ends on page 9, and the PDF has 57 total pages;
the separately maintained anonymous ICLR submission is unchanged. The source
ZIP includes the template's MIT license (`arxiv-style-LICENSE.txt`).

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
