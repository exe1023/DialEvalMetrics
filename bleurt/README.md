# BLEURT

We keep the installation steps from the README in original repository, please follow it to install.

## Installation

BLEURT runs in Python 3. It relies heavily on `Tensorflow` (>=1.15) and the
library `tf-slim` (>=1.1, currently only available on GitHub).
You may install it as follows:

```
pip install --upgrade pip  # ensures that pip is current
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

You may check your install with unit tests:

```
python -m unittest bleurt.score_test
python -m unittest bleurt.score_not_eager_test
python -m unittest bleurt.finetune_test
```

