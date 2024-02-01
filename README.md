# ABM-Grp5

This is a repository for the ABM group project of group 5:
- José Cunha - 5216087
- Emma Lombardo - 4464249
- Derek Warner - 4565517

## Project description
add in description of the ABM 

## How to run
To run the model, run the following command in the terminal, where the parameters are:
- p: polarization parameter
- n: number of households
- plot: whether to plot the model or not
```bash
python -m run -p 0.6 -n 100 -plot
```

To batch run the model, run the following command in the terminal:
```bash
python -m run --function run_batch
```

## Reusable Building Block
Can find the code for the RBB in `model.py` and the link to the RBB description in the following link: \
https://www.agentblocks.org/admin/rbb