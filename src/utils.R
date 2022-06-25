library(stringr)

shape = function(x){
  return(c(nrow(x),ncol(x)))
}


prime = function(x){
  stopifnot(is.numeric(x))

  res = c(2:x)
  flag = rep(TRUE,x-1)

  for (i in 2:sqrt(x)){
    flag[which((res %% i) == 0 & res > i)] = FALSE
  }

  return(res[flag])
}


sigmoid = function(x){
  return (1 / (exp(-x) + 1))
}


len = function(x){
  return(length(x))
}


syn_save_path = function(...){
  params = c(...)
  save_file = str_c(params, collapse='#')
  save_file = paste0('../../results/synthetic/mcem100rep/', save_file)
  return (save_file)
}


mst_save_path = function(...){
  params = c(...)
  save_file = str_c(params, collapse='#')
  save_file = paste0('../../results/mst/mcem100rep/', save_file)

  return (save_file)
}


proj = function(x){
  return (x %*% solve(t(x) %*% x) %*% t(x))
}