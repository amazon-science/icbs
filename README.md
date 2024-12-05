[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/downloads/release/python-3100/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0

# iCBS: Iterative Combinatorial Brain Surgeon

iCBS is a research library implementing a scalable optimization approach to pruning
neural networks by using block coordinate descent.

## In more detail

Pruning neural networks, which involves removing a fraction of their weights, can often
maintain high accuracy while significantly reducing model complexity, at least up to a
certain limit. We present a neural network pruning technique that builds upon the
Combinatorial Brain Surgeon, but solves an optimization problem over a subset of the
network weights in an iterative, block-wise manner using block coordinate descent. The
iterative, block-based nature of this pruning technique, which we dub ``iterative
Combinatorial Brain Surgeon'' (iCBS) allows for scalability to very large models,
including large language models (LLMs), that may not be feasible with a one-shot
combinatorial optimization approach. When applied to large models like Mistral and DeiT,
iCBS achieves higher performance metrics at the same density levels compared to existing
pruning methods such as Wanda. This demonstrates the effectiveness of this iterative,
block-wise pruning method in compressing and optimizing the performance of large deep
learning models, even while optimizing over only a small fraction of the weights.
Moreover, our approach allows for a quality-time (or cost) tradeoff that is not
available when using a one-shot pruning technique alone. The block-wise formulation of
the optimization problem enables the use of hardware accelerators, potentially
offsetting the increased computational costs compared to one-shot pruning methods like
Wanda. In particular, the optimization problem solved for each block is quantum-amenable
in that it could, in principle, be solved by a quantum computer.

iCBS was developed by the Amazon Quantum Solutions Lab, the FCAT Quantum and Future
Computing Incubator, and the AI Center of Excellence at Fidelity Investments. For more
information see our [scientific paper](https://arxiv.org/abs/2411.17796).

## Getting started

For a simple example of pruning a classifier, check out the Jupyter notebook at
`notebooks/simple_example.ipynb`. For a simple example of pruning an LLM see
`notebooks/simple_example_llm.ipynb`.

## Installation

### icbs

This package requires Python 3.9 or later, but using Python 3.10 or later is
recommended. To install this package and its requirements (see `requirements.txt`) in
editable mode, execute the following command in the package root directory (we highly
recommend using a virtual environment):

`pip install -r requirements.txt -e .`

In addition, please follow the compilation instructions for the `sa-card` package
located in `src/sa-card`. Then copy the compiled executable `sa-card` to the
`src/icbs` folder.

### LM Evaluation Harness

In order to run the example notebooks, the EleutherAI language model evaluation harness
needs to be installed:

```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Running the tests

To run all the tests execute the following command in the `tests/` directory:

`pytest tests`

## Contributing

Contributions are welcome!

Please note:

- We use [`black`](https://github.com/psf/black) for formatting, with the default line
  length of 88 characters.
- Docstrings and naming should follow the [Google Python Style
  Guide](https://google.github.io/styleguide/pyguide.html).
- Contributions should follow the pull request workflow and be reviewed by at least one
  person.
- Consider adding tests to your contribution.
- Always run the tests before making a pull request.

## Citation

If you use iCBS in a publication, please cite it as:

```bibtex
@article{icbs2024,
  title={Scalable iterative pruning of large language and vision models using block coordinate descent},
  author={Rosenberg, Gili and Brubaker, J Kyle and Schuetz, Martin JA and Zhu, Elton Yechao and Kad{\i}o{\u{g}}lu, Serdar and Borujeni, Sima E and Katzgraber, Helmut G},
  journal={arXiv preprint arXiv:2411.17796},
  year={2024}
}
```

## License

iCBS is licensed under the [CC BY-NC 4.0](LICENSE) non-commercial license.
<br>
