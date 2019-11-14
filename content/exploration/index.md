---
layout: post
title: Exploration
---

## Sketch generation

For testing purposes, one might run the script standalone, as:

`python sketch_generation.py <code_snippet>`

for example:

```
$ python sketch_generation.py "xs = [f(x) for f, x in self.get('func', 10)]"
NAME = [ FUNC#1 ( NAME ) for FUNC#1 , NAME in self . FUNC#2 ( STRING , NUMBER ) ]
```

Using it in a pre-processing stage:

```python
from sketch_generation import Sketch
...
code: str = ... # e.g. "ps = filter(primes, xs)"
sketch = Sketch(code, verbose).generate()
print(sketch)
```

### Algorithm

- keep reserved Python keywords as is
- strip off arguments and variable names
- substitute literals with their types: `NUMBER` or `STRING`
- specialize `NAME` token for functions: `FUNC#<arity>`; done by traversing the AST generated for the code snippet and querying nodes of type `ast.Call`: name and `len(args)` is extracted.

**Examples**

```
x = 1 if True else 0
NAME = NUMBER if True else NUMBER

result = SomeFunc(1, 2, 'y', arg)
NAME = FUNC#4 ( NUMBER , NUMBER , STRING , NAME )

result = [x for x in DoWork(xs) if x % 2 == 0]
NAME = [ NAME for NAME in FUNC#1 ( NAME ) if NAME % NUMBER == NUMBER ]
```

---

## Fine-tune word embeddings

The fine-tune process makes use of the [mittens](https://github.com/roamanalytics/mittens) framework.
{% include sidenote.html id="note-glove" note="For the moment, only GloVe embeddings are considered."%}
Mittens extends GloVe in a retrofitting model, by adding a regularization term $$\mu$$, which penalizes the distance between
an original ($$\widehat{w_i} = w_i + \tilde{w_i}$$) and a new ($$r_i$$) word vector,
with the goal of keeping the new embedding close to the original one:

$$J_{Mittens} = J_{GloVe} + \mu \sum_{i \in V} \left \lVert \widehat{w_i} - r_i \right \rVert ^2$$

where

$$J_{GloVe} = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T\tilde{w_j} + b_i + \tilde{b_j} - \log X_{ij})^2$$

The fine-tune algorithm proceeds to construct a vocabulary of the top $$N$$ most frequent words in the given corpora,
ignoring words with a frequency $$< f$$, and those that are not already present in the pre-trained embedding matrix (if specified).
The vocabulary size is thus $$V \leq N$$. Afterwards, a co-occurrence matrix of size $$V \times V$$ is constructed by scanning all considered words in a given window of specified size.
Finally, after this pre-processing part, the mittens module is invoked, performing $$T$$ steps of fine-tuning, with the specified co-occurrence matrix, vocabulary, pre-trained matrix and regularization value $$\mu$$.
After having constructed the fine-tuned vector, the final fine-tuned embedding space is computed as such:

$$w = \alpha \cdot w_{orig} + \beta \cdot w_{ft}$$

where $$w$$ is the combined word vector, $$w_{orig}$$ is the pre-trained word vector, $$w_{ft}$$ is the fine-tuned word vector, and $$\alpha, \beta$$ are weighting constants, with $$\alpha + \beta = 1$$.
{% include sidenote.html id="note-ft" note="In most of our experiments, we have found that the best performance is achieved with $$\alpha \in [0.05, 0.2]$$ and $$\beta \in [0.8, 0.95]$$. Also, after some exploratory data analysis, we have considered the top $$8,000$$ most frequent words in the corpus, window size $$\in \{5, 7\}$$, regularization factor $$\mu = 0.1$$, training for $$10,000$$ iterations." %}

### Usage

Arguments:

```
-root_dir    # root directory used to store files
-data_source # path to dir with files OR file listing (for corpus)
-exp_name    # experiment name (%Y-%m-%d_%H-%M-%S timestamp will be automatically prepended)
-pt_emb_file # pre-trained embeddings file
-num_ft_iter # number of fine-tuning iterations (default = 1000)
-vocab_size  # number of unique words to consider (at most!) (default = 20000)
-window_size # consider words in window (w_i-N ... w_i ... w_i+N) (default = 5)
-min_freq    # consider words with frequency >= min_freq (default = 1)
-mu          # regularization factor (mu from mittens paper) (default = 0.1)
-only_in_emb # if true, only use words that already exist in the pre-trained embeddings
```

For example, fine-tuning GloVe (`glove.6B.200d.txt`) on Python questions from StackOverflow results in an experiment
folder named `2019-05-14_18-19-19-python-so-200`, containing:

```
2019-05-14_18-19-19-python-so-200
├── config.txt      # dump of fine-tune settings
├── python-so.emb   # fine-tuned embeddings (mittens output): (V x emb_dim) float32 numpy array
├── python-so.mat   # co-occurrence matrix: (V x V) uint16 numpy array
└── python-so.vocab # vocabulary: Dict[str, int]
```

---

## Data pre-processing

Pre-processing scripts will generate train / dev / test splits for the dataset.
This stage also includes sketch generation and input sanitization.

Output format follows Coarse-to-Fine model: token = actual code, type = sketch, src = intent.

### Django

Arguments:
```
-in_dir     # input directory containing all.code and all.anno files
-out_dir    # output directory where {train, dev, test}.json files will be dumped
-dev_split  # % of all examples to use for validation (default = 0.05)
-test_split # % of all examples to use for testing (default = 0.1)
```

Sanitization consists of:
- replacing all string literals with `_STR:<n>_` where `n` is the index of the string in the intent (zero-based)
- removing stray / weird chars from the intent

Final training example:
```
{
    "token": ["decimal_separator", "=", "get_format", "(", "\" _STR:0_ \"", ")"],
    "type" : ["NAME", "=", "FUNC#1", "(", "STRING", ")"],
    "src"  : ["call", "the", "function", "get_format", "with", "an", "argument", "string", "_STR:0_", "substitute", "the", "result", "for", "decimal_separator"]
}

```

### CoNaLa

CoNaLa pre-processing is similar to Django's, but without `_STR:<n>_` markers.

Final training example:
```
{
    "token": ["[", "x", "for", "x", "in", "file", ".", "namelist", "(", ")", "if", "x", ".", "endswith", "(", "'/'", ")", "]"],
    "type" : ["[", "NAME", "for", "NAME", "in", "NAME", ".", "FUNC#0", "(", ")", "if", "NAME", ".", "FUNC#1", "(", "STRING", ")", "]"]
    "src"  : ["list", "folders", "in", "zip", "file", "'file'", "that", "ends", "with", "'/'"],
}
```

---

## Mining docstrings

---

## Code equivalence
