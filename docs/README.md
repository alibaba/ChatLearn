# Setup

```
apt-get install latexmk texlive-xetex fonts-noto fonts-freefont-otf xindy latex-cjk-all
pip install -r requirements.txt
```

# build pdf

```
# cd en for english doc
cd zh
# make latexpdf
sphinx-build -b latex . _build/latex
cd _build/latex

# modify chatlearn.tex for auto wrap of text in the table
# Find the table with `stream\\_data\\_loader\\_type`, replace `\begin{tabulary}{\linewidth}[t]{TTT}` with `\begin{tabularx}{\linewidth}[t]{|l|l|X|}`
# and replace the corresponding `\end`
# save the change, and execute
make all-pdf
```


# build html
```
# cd en for english doc
cd zh
make html
```

