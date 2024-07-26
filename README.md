# DySMHO_higherDim

# System Requirements

Software requirements and versions tested
- Python 3.7.13
- [Pyomo](http://www.pyomo.org/) 6.1.2
- [numpy](https://numpy.org/) 1.19.5
- [scipy](https://www.scipy.org/) 1.6.2
- [scikit-learn](https://scikit-learn.org/) 0.24.2
- [statsmodels](https://www.statsmodels.org/stable/index.html) 0.12.2
- [matplotlib](https://matplotlib.org/) 3.4.3
- [cvxopt](https://cvxopt.org/) 1.3.2
- [GAMS](https://www.gams.com/)
	- [CONOPT](http://www.conopt.com/) Nonlinear solver (license is required for large instances)

# Installation

1. Clone the repository

Run the following command in the terminal:
```
git clone https://github.com/MasahitoUWAMICHI/DySMHO_higherDim.git
```

2. Install GAMS

Please follow the instructions on the GAMS website to install GAMS on your system.

3. Install the packages

Run the following command in the terminal:
```
pip install .
```

4. Set the path

Run the following command in the terminal:
```
python setup.py develop
```


# Reference

This repository includes modifications from the following repository by Baldea-Group:
https://github.com/Baldea-Group/DySMHO

# License

MIT License

Copyright (c) 2024 Masahito Uwamichi

This software is a modified version of the original software licensed under the MIT License. Original copyright (c) 2021 Baldea-Group. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
