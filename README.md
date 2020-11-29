# NBA machine learning project

![SacTown](https://e0.365dm.com/18/12/768x432/deaaron-fox-sacramento-kings_4531686.jpg)

### The goal of this project is to predict the winner of any NBA game by engineering

### at least 50 features deployed into a random forest model, accessible through docker

# Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`

# Run `pre-commit` locally.

`pre-commit run --all-files`
