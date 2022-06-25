# Estimating Three-and Four-parameter MIRT Models with Importance-weighted Sampling Enhanced Variational Autoencoder

Codes for reproducing synthetic experiments in [Estimating Three-and Four-parameter MIRT Models with Importance-weighted Sampling Enhanced Variational Autoencoder](https://www.frontiersin.org/articles/10.3389/fpsyg.2022.935419/abstract)

### Get started

To install the package, run

```
$ git clone https://github.com/lliutianc/mirt-iwvae
$ cd mirt-iwvae
# It is recommended to create a virtual environment here
$ pip install -e .
$ pip install -r requirements.txt

```

### Synthetic Experiments

Below are our bash scripts to fit IWVAE and MCEM on high performance computing cluster. 

```
    # Fit IWVAE
    for asymp in single double; do
        for pl in 3 4; do
            for depend in 1 2; do 
                for corr_factor in 0 1; do 
                    echo "Start IWVAE: Asymptotic: "$asymp", MIRT: "$pl", Depend: "$depend", Correlated Factors: "$corr_factor
                    python fit_iwvae_syn.py -asymptotic $asymp \
                                            -pl $pl \
                                            -item_depend $depend \
                                            -correlated_factor $corr_factor \
                                            -replication_id $SLURM_ARRAY_TASK_ID
                done
            done 
        done
    done

    # Fit MCEM
    for asymp in single double; do
        for pl in 3 4; do
            for depend in 1 2; do 
                for corr_factor in 0 1; do 
                    echo "Start MCEM: Asymptotic: "$asymp", MIRT: "$pl", Depend: "$depend", Correlated Factors: "$corr_factor
                    Rscript --vanilla fit_mcem_syn.R $asymp \
                                                    $pl \
                                                    $depend \
                                                    $corr_factor \
                                                    $SLURM_ARRAY_TASK_ID
                done
            done 
        done
    done
```

where `SLURM_ARRAY_TASK_ID` ranges from 1 to 100, indicates 100 independent replications. 