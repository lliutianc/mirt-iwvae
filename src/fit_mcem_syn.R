#! ~/miniconda3/bin/Rscript

library(stringr)
library(mirt)
library(pracma)
library(matrixcalc)
library(combinat)

# python configuration
library(reticulate)
use_python('~/miniconda3/bin/python3', required=T)  ## Replace with your Python env 
py_config()
py_available()
source_python('data.py')
source_python('evaluation.py')

source('utils.R')


mcem_trial = function(
    n, 
    j, 
    k, 
    pl, 
    with_b=1,
    correlated_factor=False,
    a_dist='uniform', 
    a_shape='s',
    factor_influ=5, 
    item_depend=1,
    max_observed=NULL, 
    svd_init=0,
    mirt_method='EM', 
    max_iter=max_iter,
    replication_id=1, 
    save_path=NULL){

    if (is.null(max_observed)) max_observed = j %/% 5

    data = load_data(
        n, 
        j, 
        k, 
        pl, 
        with_b=with_b, 
        a_dist=a_dist, 
        a_shape=a_shape,
        svd_init=svd_init,
        factor_influ=factor_influ,
        item_depend=item_depend,
        max_observed=max_observed,
        correlated_factor=correlated_factor,
        seed=1)

    results = list()
    results[['data']] = data

    Y = data[[1]]
    Y[is.nan(Y)] = NA
    Y = data.frame(Y)

    if (pl == 2) itemtype = '2PL'
    if (pl == 3) itemtype = '3PL'
    if (pl == 4) itemtype = '4PL'
    
    time = proc.time()
    mirt_result = try(
        mirt(
            Y, 
            model=k, 
            itemtype=itemtype, 
            method=mirt_method,
            GenRandomPars=TRUE, 
            rotate='none',
            technical=list(NCYCLES=1000),),
        silent=TRUE)

    if ('try-error' %in% class(mirt_result)){
        fail = TRUE
    } else{
        fail = FALSE
        time_cost = as.numeric((proc.time() - time)[3])
        print(time_cost)
        params = coef(mirt_result, simplify=1, rotate='none')[[1]]
        pred_a = params[, 1:k]
        pred_b = params[, (k+1)]
        if (pl >= 3){
            pred_c = params[, (k+2)]
        } else{
            pred_c = rep(0, j)
        }

        if (pl >= 4){
            pred_d = params[, (k+3)]
        } else{
            pred_d = rep(1, j)
        }

        qmc = 0
        if (k > 3) qmc = 1
        pred_x = fscores(mirt_result, rotate='none', QMC=qmc)
        
        results[['mirt']] = mirt_result
        results[['fit_time']] = time_cost

        results[['pred_x']] = pred_x
        results[['pred_a']] = pred_a
        results[['pred_b']] = pred_b
        results[['pred_c']] = pred_c
        results[['pred_d']] = pred_d

        if (!is.null(save_path)){
            write.csv(pred_a, paste0(save_path, '_a.csv'), row.names=FALSE)
            write.csv(pred_b, paste0(save_path, '_b.csv'), row.names=FALSE)
            write.csv(pred_c, paste0(save_path, '_c.csv'), row.names=FALSE)
            write.csv(pred_d, paste0(save_path, '_d.csv'), row.names=FALSE)
            write.csv(pred_x, paste0(save_path, '_factors.csv'), row.names=FALSE)
            write.csv(time_cost, paste0(save_path, '_time.csv'), row.names=FALSE)
        }
    }

    results[['is_fail']] = fail

    return (results)
}


is_succ_trial = function(to_save, pl){
    if (pl == 3){
        to_check_file = c('_a.csv', '_b.csv', '_c.csv', '_time.csv', '_factors.csv')
    } else{
        to_check_file = c('_a.csv', '_b.csv', '_c.csv', '_d', '_time.csv', '_factors.csv')
    }
    
    is_succ = TRUE
    for (file in to_check_file){
        to_check = paste0(to_save, file)
        is_succ = is_succ & file.exists(to_check)
    }

    return (is_succ)
}


args = commandArgs(trailingOnly=TRUE)

mirt_method = 'MCEM'
if (length(args)==5){
    asymptotic = args[1]
    pl = args[2]
    item_depend = args[3]
    correlated_factor = args[4]
    replication_id = args[5]
} else{
    stop('Unknown paramters')
}

train_new = TRUE
correlated_factor = as.logical(as.numeric(correlated_factor))
if (correlated_factor==FALSE){
    factor_cov = 'diagonal'
} else{
    factor_cov = 'correlated'
}

rotate_method = 'promax'

# hyper-parameters
k = 5
pl = as.numeric(pl)
item_depend = as.numeric(item_depend)

factor_influ = 5
a_dist = 'uniform'
a_shape = 's'
svd_init = FALSE
max_iter = 500

ns = c(500, 1000, 5000, 10000)

if (asymptotic=='double'){
    js = c(100, 200, 300, 500)
} else{
    js = c(100, 100, 100, 100)
}

for (idx in 1:length(ns)){
    n = ns[idx]
    j = js[idx]
    max_observed = j %/% 5

    to_save = syn_save_path(
        factor_cov,
        n, 
        j, 
        k, 
        pl,
        a_dist,
        a_shape,
        factor_influ,
        item_depend,
        max_observed,
        svd_init)

    to_save = paste0(to_save, '_rep', replication_id)
    if ((!is_succ_trial(to_save, pl)) | train_new){

        writeLines(paste('Start: trial_path: ', to_save))
        fit = mcem_trial(
            n, 
            j, 
            k, 
            pl, 
            correlated_factor=correlated_factor,
            a_dist=a_dist, 
            a_shape=a_shape,
            factor_influ=factor_influ, 
            item_depend=item_depend,
            max_observed=max_observed, 
            svd_init=svd_init,
            mirt_method=mirt_method, 
            max_iter=max_iter,
            replication_id=replication_id, 
            save_path=to_save)

        saveRDS(fit, paste0(to_save, '.rds'))
        is_fail = fit[['is_fail']]
        if (is_fail == TRUE){
            writeLines(paste('Fail: trial_path: ', to_save))
        } else{
            writeLines(paste('Success: trial_path: ', to_save))
        }

    } else{
        writeLines(paste('Existing: trial_path: ', to_save))

        # tmp: save time
        # writeLines('Extract fit time...')
        # fit = readRDS(paste0(to_save, '.RData'))
        # time_cost = fit[['time_cost']]
        # write.csv(time_cost, paste0(to_save, '_time.csv'), row.names=FALSE)
    }
   
}