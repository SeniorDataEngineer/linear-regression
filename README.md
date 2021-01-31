# <a name="top-page"></a>LinearRegressionPython
> Is an implementation of linear regression written
from pseudo-code. The project includes a Jupyter
Notebook illustrating usage of the package.

# Table of Contents
* [License](#license)
* [Team Members](#team-members)
* [Features](#features)
* [Project Structure](#structure)
* [Standards](#standards)
* [Testing](#testing)

# <a name="license"></a>License
* None

# <a name="team-members"></a>Team Members
* "Mr. James Rose" <jameserose8@gmail.com>

# <a name="features"></a>Features
* update this!


# <a name="structure"></a>Project Structure
```
|_ Notebooks
  |_ linear_regression_demo.ipynb 
|_ src  
  |_ my_linear_regression
    |_ mixin
      |_ __init__.py 
      |_ has_property_mixin.py
    |_ neuron
      |_ __init__.py 
      |_ artificial_neuron.py
    |_ __init__.py 
    |_ linear_regression.py
  |_setup.py
|_ .gitignore
|_ README.md
|_ requirements.txt
```

# <a name="standards"></a>Standards
All code has been checked with `pycodestyle` utility to match  
**pep8** guidance. 
## Usage of `pyscodestyle`
Navigate to directory where the file you want to test is located  
using `cd` in terminal. Run `pycodestyle filename.py` and wait  
for output in terminal. If no output standard is met.

# <a name="testing"></a>Testing
## Unit testing
The unit tests for the package are stored in the python modules
as doctests. When running a module standalone the doctests will
be called for that module.  
To run tests; navigate to *src/my_k_means/* and run command  
`python -m filename.py`.

-------
[Return to top](#top-page)