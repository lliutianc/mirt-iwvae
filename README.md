# Estimating Three-and Four-parameter MIRT Models with Importance-weighted Sampling Enhanced Variational Autoencoder

Codes for reproducing synthetic experiments in [Estimating Three-and Four-parameter MIRT Models with Importance-weighted Sampling Enhanced Variational Autoencoder](https://www.frontiersin.org/articles/10.3389/fpsyg.2022.935419/abstract)

### Prerequisites

Our experiments are based on PYTHON=3.8, 
see `requirements.txt` for required packages. 

### Synthetic Experiments

Below are bash scripts to fit IWVAE and MCEM on high performance computing cluster. 

```
    # Fit IWVAE
    for asymp in single double; do
        for pl in 3 4; do
            for depend in 1 2; do 
                for corr_factor in 0 1; do 
                    echo "Start IWVAE: Asymptotic: "$asymp", MIRT: "$pl", Depend: "$depend", Correlated Factors: "$corr_factor
                    python fit_iwvae_syn.py -asymptotic $asymp \
                                            -pl $pl \
                                            -item-depend $depend \
                                            -correlated-factor $corr_factor \
                                            -replication-id $SLURM_ARRAY_TASK_ID
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