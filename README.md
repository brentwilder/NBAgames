# NBA machine learning project

![SacTown](https://e0.365dm.com/18/12/768x432/deaaron-fox-sacramento-kings_4531686.jpg)

### The goal of this project was to predict the winner of any NBA game by feature engineering a machine learning (classification) model. Model exploration was conducted and reports were generated in this repository. Finally, this model was then saved to a pkl file and made accessible through docker.

## Clone this repository and open it up in your favorite IDE (VS Code <3)

`git clone https://github.com/bwilder95/NBAgames.git`

## Run docker to initiate the model

`docker-compose up`

NOTE: As a disclaimer for those that run the docker container, this process will take a long time to load in all the python requirements. Further, the python code utilized all available classification models and used brute force to find all model combinations. With that being said, current run time for this entire process is about an hour to an hour and a half. You should see messages appear that can give you an idea of what stage of the code you are on (you will have to be a little patient here). The longest process is the brute force method using mlxtend Sequential Feature Selector, as this package assesses all feature combinations. The code will pick the best combination and run the finalized model automatically.

# Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`

## Run `pre-commit` locally.

`pre-commit run --all-files`
