---
layout: post
title: Data Exploration
---

# 1. Sketch generation

### Usage

Sketch generation is done by the `Sketch` class in
[`preprocess/sketch_generation.py`](https://github.com/code-gen/data-exploration/blob/master/preprocess/sketch_generation.py).
For testing purposes, one might run the script standalone, as:

`python sketch_generation.py <code_snippet>`

for example:

`python sketch_generation.py "xs = [f(x) for f, x in self.get('func', 10)]"`

Usage in pre-processing stage:

```python
from sketch_generation import Sketch
...
code: str = ... # e.g. "ps = filter(primes, xs)"
sketch = Sketch(code, verbose).generate()
print(sketch)
```

### Algorithm

- keep Python keywords as is
- strip off arguments and variable names
- substitute literals with their types: `NUMBER` or `STRING`
- specialize `NAME` token for functions: `FUNC#<arity>`

Refining `NAME` to `FUNC#<arity>` is done by traversing the AST generated for the code snippet
and querying nodes of type `ast.Call`: name and `len(args)` is extracted.

**Examples**

```
x = 1 if True else 0
NAME = NUMBER if True else NUMBER

result = SomeFunc(1, 2, 'y', arg)
NAME = FUNC#4 ( NUMBER , NUMBER , STRING , NAME )

result = [x for x in DoWork(xs) if x % 2 == 0]
NAME = [ NAME for NAME in FUNC#1 ( NAME ) if NAME % NUMBER == NUMBER ]
```

# 2. Fine-tuning word embeddings

The fine-tune process makes use of the [mittens](https://github.com/roamanalytics/mittens) framework.
{% include sidenote.html id="note-glove" note="For the moment, only GloVe embeddings are considered."%}
Mittens extends GloVe in a retrofitting model, by adding a regularization term $$\mu$$, which penalizes the distance between
an original ($$\widehat{w_i} = w_i + \tilde{w_i}$$) and a new word vector ($$r_i$$),
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
{% include sidenote.html id="note-ft" note="In most of our experiments, we have found that the best performance is achieved with $$\alpha \in [0.05, 0.2]$$ and $$\beta \in [0.8, 0.95]$$. Also, after some exploratory data analysis, we have considered the top $$8,000$$ most frequent words in the corpus, the window size $$\in \{5, 7\}$$, regularization factor $$\mu = 0.1$$, and training for $$10,000$$ iterations." %}

### Usage

The script that performs the fine-tuning is [`emb_fine_tune/glove_fine_tune.py`](https://github.com/code-gen/data-exploration/blob/master/emb_fine_tune/glove_fine_tune.py).
One may run it using a bash script, for example:

```bash
base_dir=$(dirname "$(readlink -f $0)")

python emb_fine_tune/glove_fine_tune.py \
    -root_dir ${base_dir}/../embeddings \
    -data_source ${base_dir}/../corpus/python-stackoverflow/question_words_clean.pickle \
    -exp_name python-so \
    -pt_emb_file ${base_dir}/../embeddings/glove.6B.200d.txt \
    -num_ft_iter 10000 \
    -vocab_size 10000 \
    -window_size 7 \
    -min_freq 10 \
    -mu 0.1 \
    -only_in_emb
```

Arguments are:

```
-root_dir       ROOT_DIR        Root directory used to store files
-data_source    DATA_SOURCE     Path to dir with files OR file listing (for corpus)
-exp_name       EXP_NAME        Name for current experiment (%Y-%m-%d_%H-%M-%S timestamp will be automatically prepended)
-pt_emb_file    PT_EMB_FILE     Pre-trained embeddings file
-num_ft_iter    NUM_FT_ITER     Number of fine-tuning iterations
-vocab_size     VOCAB_SIZE      Number of unique words to consider (at most!)
-window_size    WINDOW_SIZE
-min_freq       MIN_FREQ        Consider words with frequency >= min_freq
-mu             MU              Regularization factor (mu from mittens paper) (default = 0.1)
-only_in_emb    If true, only use words that already exist in the pre-trained embeddings
```

For example, fine-tuning GloVe (`glove.6B.200d.txt`) on Python questions from StackOverflow results in an experiment
folder named `2019-05-14_18-19-19-python-so-200`, containing:

```
2019-05-14_18-19-19-python-so-200
├── config.txt          -- dump of fine-tune settings
├── python-so.emb       -- fine-tuned embeddings (mittens' output)
├── python-so.mat       -- co-occurrence matrix: N x N numpy array, dtype=uint16
└── python-so.vocab     -- vocabulary: Dict[str, int]
```

# 3. Data pre-processing

### Django


### CoNaLa
