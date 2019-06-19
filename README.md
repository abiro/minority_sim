# Minority Rule Simulation

Run Monte Carlo simulations of the effects of the minority rule on payment 
protocol adoption. Read more of the background in [this post](https://agost.blog/2019/06/mapping-crypto-minority-rule/) or check out
the [notebook](notebook.ipynb) for an explanation on methodology and 
results.

## How to run

Requires Python 3.5 or later. It is recommended to install dependencies in a 
virtual environment.

1. Install dependencies: `pip install -r requirements.txt`
1. Run the simulation with default configs and save the results in the current
directory: `python minority_sim.py`

## Custom configurations

To run a simulation with custom configuration:

1. Create a copy of [config.yaml](config.yaml) and adjust the parameters.
1. Pass your config file to the script with: 
`python minority_sim.py --config=./my_config.yaml`

Run `python minority_sim.py --help` for additional options.
