[![PyPI - Version](https://img.shields.io/pypi/v/scene-synthesizer)](https://pypi.org/project/scene-synthesizer/)
[![Static Badge](https://img.shields.io/badge/docs-passing-brightgreen)](https://scene-synthesizer.github.io/)
[![Static Badge](https://img.shields.io/badge/arXiv-preprint-D12424)](https://drive.google.com/file/d/1fewL5ezXhlICAv_BNqyLe5YjaXAbXH2Y/view?usp=sharing)

# scene_synthesizer

A python package to generate manipulation scenes.
Check the [documentation](https://scene-synthesizer.github.io/) page for detailed installation instructions, API, and examples.

## Installation

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use (e.g. [OpenUSD license](https://openusd.org/license>)).

### Via pip
```
pip install scene-synthesizer[recommend]
```

### Via git
```
# create and activate conda env or venv
git clone https://github.com/NVlabs/scene_synthesizer.git
cd scene_synthesizer/
pip install -e.[recommend]
```
See the [documentation](https://scene-synthesizer.github.io/getting_started/install.html) for detailed installation instructions.

## License

The code is released under the [Apache-2.0 license](https://github.com/NVlabs/scene_synthesizer/blob/main/LICENSE).

## How can I cite this library?

```
@article{Eppner2024, 
    title = {scene_synthesizer: A Python Library for Procedural Scene Generation in Robot Manipulation}, 
    author = {Clemens Eppner and Adithyavairavan Murali and Caelan Garrett and Rowland O'Flaherty and Tucker Hermans and Wei Yang and Dieter Fox},
    journal = {Journal of Open Source Software}
    publisher = {The Open Journal}, 
    year = {2024},
}
```
