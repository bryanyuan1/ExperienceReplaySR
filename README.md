# Supplementary Material

This is the code base for our paper <i>Improving Experience Replay with Successor Representation</i>. There are three parts each corresponding to one algorithm: Prioritized Sweeping with Successor Representation (PS-SR), tabular PER with SR (Vanilla PER-SR), and original PER with SR (PER-SR).

## Prioritized Sweeping with SR (PS-SR)
We experiment our new version of PS-SR using the Dyna Maze experiment used by Sutton and Barto. 
- run `PS_dynamaze.ipynb` to start the experiment and see the plot

## Tabular PER with SR (Vanilla PER-SR)
We replicate the experiment of the vanilla PER without neural networks, which was done in the PER paper. 

- run `vanilla_PER_cliffwalk.ipynb` to run Cliffwalk experiment and see the plot

## PER with SR (PER-SR)

For PER-SR, we use the Atari benchmarks. 

- **atari_plots.ipynb**: the jupyter notebook to generate plots from data in folder `running-data`. 
- **train_dopamine_per.py**: the file for running 200 iterations of a specific Atari game using PER, save data to `running-data/prioritized_dqn/GAME` for the game specified in variable `GAME`.
- **train_dopamine_norm.py**: similar to **train_dopamine_per.py**, runs the PER-SR algorithm.

To run the Atari experiments, follow the steps below:
- Follow instructions [here](https://github.com/google/dopamine) to install `dopamine`. 
- install the Atari game ROMs `!python -m atari_py.import_roms roms`
- Change the `GAME` variable in file `train_dopamine_norm.py` to the desired Atari game.
- Run command `python train_dopamine_norm.py`. Similar for `train_dopamine_per.py`. 