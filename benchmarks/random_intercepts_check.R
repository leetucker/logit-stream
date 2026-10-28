# Fits for benchmarks/random_intercepts_check.py: Rscript random_intercepts_check.R DATA OUTDIR {binary|ordered|multinomial}
suppressMessages({library(lme4); library(ordinal); library(mclogit); library(arrow)})
args <- commandArgs(trailingOnly = TRUE)
d <- as.data.frame(read_parquet(args[1])); out <- args[2]; what <- args[3]
d$g <- factor(d$g)
res <- list()
t0 <- Sys.time()
if (what == "binary") {
  for (q in c(0, 1, 25)) {
    t <- Sys.time()
    m <- glmer(y ~ x1 + x2 + (1 | g), data = d, family = binomial, nAGQ = q,
               control = glmerControl(optimizer = "bobyqa"))
    res[[paste0("nAGQ", q)]] <- c(fixef(m), sd = attr(VarCorr(m)$g, "stddev"),
                                   se = sqrt(diag(as.matrix(vcov(m)))), loglik = as.numeric(logLik(m)),
                                   seconds = as.numeric(difftime(Sys.time(), t, units = "secs")))
  }
} else if (what == "ordered") {
  d$y <- factor(d$y, ordered = TRUE)
  for (q in c(1, 10)) {
    t <- Sys.time()
    m <- clmm(y ~ x1 + x2 + (1 | g), data = d, link = "logit", nAGQ = q)
    res[[paste0("nAGQ", q)]] <- c(m$beta, m$alpha, sd = sqrt(as.numeric(VarCorr(m)$g)),
                                   se = sqrt(diag(vcov(m)))[names(m$beta)], loglik = as.numeric(logLik(m)),
                                   seconds = as.numeric(difftime(Sys.time(), t, units = "secs")))
  }
} else {
  d$y <- factor(d$y)
  t <- Sys.time()
  m <- mblogit(y ~ x1 + x2, random = ~ 1 | g, data = d, method = "PQL", trace = FALSE)
  res[["PQL"]] <- c(coef(m), Sigma = as.numeric(m$VarCov[[1]]),
                    seconds = as.numeric(difftime(Sys.time(), t, units = "secs")))
}
for (k in names(res)) write.csv(data.frame(name = names(res[[k]]), value = unname(res[[k]])),
                               file.path(out, paste0(what, "_", k, ".csv")), row.names = FALSE)
