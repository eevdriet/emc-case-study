# Seminar Case Study: EMC

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Group](#group)

## Installation
```bash
# 1 Retrieve the code base
git clone https://github.com/eevdriet/emc-case-study
cd emc-case-study

# 2 Create virtual environment to install packages into
python -m venv venv
source venv/bin/activate
pip install -e .                    # internal package
pip install -r requirements.txt     # external packages

# 3 Make sure the data folder is complete (or generated, see below)

# 4 Run specific scripts
python src/emc/data/policy_manager.py
```

## Usage
### Code base
The `emc` package is divided into the following modules:
- `model` contains the basic classes which are used throughout the entire project, such as `Simulation` and `Policy`
- `data` contains various classes which are able to derive results from the data, for example
    - `policy_manager` finds the optimal policy for a given scenario based on some parameters
    - `level_builder` derives the expected infection level for a given bucket size for all scenarios
    Each of these modules can independently be run (unlike the modules in `model`)
- `regressors` contains the different types of regressors that were used for deriving optimal policies

Additionaly, the `log` and `util` are all-purpose modules for logging and various small code snippets

### Scripts
The modules in `scripts` are used for various purposes, such as
- Reworking the .RDS files into usable CSV and JSON data for the Python code base
- Collecting statistics from the data
- Generating figures/tables from the data

To generate the `data` that is used in the code base 'from scratch', the following steps can be taken:
1. Run the `load_*.R` scripts to generate CSV data from the .RDS files
2. Run the `scripts/pipeline.py` module.
This script sequentially reworks the CSV files to the final data formats that are used in the code.

## Group
- **Alexander Neutel** - [alexanderthn](https://github.com/alexanderthn)
- **Eertze van Riet** - [eevdriet](https://github.com/eevdriet)
- **Marinthe de Vries** - [MarinthedeVries](https://github.com/MarinthedeVries)
- **Matthijs de Keijzer** - [MRDekeijzer](https://github.com/MRDekeijzer)
