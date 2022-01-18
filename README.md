# Supplementary Material

This is the code base for our paper <i>Improving Experience Replay with Successor Representation</i>.

## Files
- **plots.ipynb**: the jupyter notebook to generate plots from data in folder `running-data`. 
- **train_dopamine_per.py**: the file for running 200 iterations of a specific Atari game using PER, save data to `running-data/prioritized_dqn/GAME` for the game specified in variable `GAME`.
- **train_dopamine_norm.py**: similar to **train_dopamine_per.py**, runs the PER-SR algorithm.

## Running
- Follow instructions [here](https://github.com/google/dopamine) to install `dopamine`. 
- install the Atari game ROMs `!python -m atari_py.import_roms roms`
- Change the `GAME` variable in file `train_dopamine_norm.py` to the desired Atari game.
- Run command `python train_dopamine_norm.py`. Similar for `train_dopamine_per.py`. 