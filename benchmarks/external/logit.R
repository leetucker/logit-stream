# One fit with an R package, for benchmarks/_harness.py: read the Parquet file
# into memory (timed as "read", as the Python libraries' reading is), fit, ask
# for the standard errors, and print one line,
#
#   RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {...}, ...}
#
# Peak memory is the process's VmHWM, as for the Python configurations. The
# coefficients are renamed to logit-stream's names (factor(x)3 -> C(x)[T.3],
# "<level>:<name>" for the multinomial logit) for the comparison.
#
#   Rscript logit.R '<spec as JSON>'     (see the benchmarks' external_command)
#
# spec$fit is one of
#   glm        stats::glm, family binomial                 spec$fml
#   feglm      fixest::feglm, binomial, iid standard errors spec$fml (may have "| fe")
#   multinom   nnet::multinom                              spec$fml, spec$levels
#   polr       MASS::polr, logistic                        spec$fml
#   clm        ordinal::clm, logit                         spec$fml
#   clogit     survival::clogit (method "exact")           spec$fml (with strata())
#   mlogit     mlogit::mlogit, conditional or nested       spec$fml, spec$nests
#   logitr     logitr::logitr, mixed logit, panel          spec$pars, spec$random, ...
#   alpaca     alpaca::feglm and biasCorr                  spec$fml
# and spec$path, spec$columns (to read), spec$threads.

spec <- jsonlite::fromJSON(commandArgs(TRUE)[1], simplifyVector = TRUE)

hwm_mb <- function() {
  line <- grep("^VmHWM:", readLines("/proc/self/status"), value = TRUE)
  as.numeric(strsplit(trimws(sub("VmHWM:", "", line)), " +")[[1]][1]) / 1024
}
now <- function() proc.time()[["elapsed"]]

# logit-stream's names: factor(x)3 -> C(x)[T.3], in every part of an interaction
py_name <- function(nm) {
  parts <- strsplit(nm, ":", fixed = TRUE)[[1]]
  parts <- sub("^factor\\(([A-Za-z0-9_.]+)\\)(.+)$", "C(\\1)[T.\\2]", parts)
  parts[parts == "(Intercept)"] <- "Intercept"
  paste(parts, collapse = ":")
}
named <- function(v) {
  out <- as.list(unname(v))
  names(out) <- vapply(names(v), py_name, "")
  out
}

suppressPackageStartupMessages(library(arrow))
baseline <- hwm_mb()
t0 <- now()
df <- as.data.frame(read_parquet(spec$path, col_select = spec$columns))
if (!is.null(spec$filter_avail) && isTRUE(spec$filter_avail)) df <- df[df$avail == 1, ]
df <- df[complete.cases(df), , drop = FALSE]
t_read <- now() - t0

fml <- if (!is.null(spec$fml)) as.formula(spec$fml) else NULL
iterations <- NA
converged <- TRUE
extra <- list()

t0 <- now()
if (spec$fit == "glm") {
  fit <- glm(fml, data = df, family = binomial())
  se <- sqrt(diag(vcov(fit)))
  coefs <- named(coef(fit))
  iterations <- fit$iter
  converged <- fit$converged
  n_obs <- nobs(fit)
} else if (spec$fit == "feglm") {
  suppressPackageStartupMessages(library(fixest))
  setFixest_nthreads(spec$threads)
  fit <- feglm(fml, data = df, family = binomial(), vcov = "iid")
  se <- se(fit)
  coefs <- named(coef(fit))
  iterations <- fit$iterations
  converged <- isTRUE(fit$convStatus)
  n_obs <- nobs(fit)
  extra$threads <- getFixest_nthreads()
} else if (spec$fit == "multinom") {
  # MaxNWts: the default (1000) is below the number of weights here; maxit:
  # the default (100) stops BFGS before convergence with 164 coefficients.
  # The standard errors (vcov) need the Hessian, which summary() computes too.
  suppressPackageStartupMessages(library(nnet))
  df[[spec$outcome]] <- factor(df[[spec$outcome]], levels = spec$levels)
  fit <- multinom(fml, data = df, maxit = 10000, MaxNWts = 100000, trace = FALSE)
  se <- sqrt(diag(vcov(fit)))
  cm <- coef(fit)
  coefs <- list()
  for (lv in rownames(cm)) for (term in colnames(cm)) {
    coefs[[paste0(lv, ":", py_name(term))]] <- cm[lv, term]
  }
  converged <- fit$convergence == 0
  n_obs <- nrow(df)
} else if (spec$fit == "polr") {
  suppressPackageStartupMessages(library(MASS))
  df[[spec$outcome]] <- factor(df[[spec$outcome]], ordered = TRUE)
  fit <- polr(fml, data = df, method = "logistic", Hess = TRUE)
  se <- sqrt(diag(vcov(fit)))
  coefs <- named(coef(fit))
  iterations <- fit$niter[[1]]
  converged <- fit$convergence == 0
  n_obs <- fit$n
} else if (spec$fit == "clm") {
  suppressPackageStartupMessages(library(ordinal))
  df[[spec$outcome]] <- factor(df[[spec$outcome]], ordered = TRUE)
  fit <- clm(fml, data = df, link = "logit")
  se <- sqrt(diag(vcov(fit)))
  coefs <- named(fit$beta)
  iterations <- fit$niter[1]
  converged <- fit$convergence$code == 0
  n_obs <- nobs(fit)
} else if (spec$fit == "clogit") {
  suppressPackageStartupMessages(library(survival))
  fit <- clogit(fml, data = df)
  se <- sqrt(diag(vcov(fit)))
  coefs <- named(coef(fit))
  iterations <- fit$iter
  n_obs <- fit$n
  extra$n_events <- fit$nevent
} else if (spec$fit == "mlogit") {
  suppressPackageStartupMessages({library(mlogit); library(dfidx)})
  df$chosen <- df$chosen == 1
  d <- dfidx(df, idx = c("case", "alt"), choice = "chosen")
  if (is.null(spec$nests)) {
    fit <- mlogit(fml, data = d)
  } else {
    # RU2 (unscaled = FALSE, the default), one dissimilarity parameter per
    # nest; a nest of one alternative has none (its iv held at 1)
    # (mlogit evaluates its arguments in a frame of its own: give it values)
    nests <- spec$nests
    held <- paste0("iv:", names(nests)[lengths(nests) == 1])
    fit <- if (length(held)) mlogit(fml, data = d, nests = nests, constPar = held)
           else mlogit(fml, data = d, nests = nests)
  }
  se <- sqrt(diag(vcov(fit)))
  co <- coef(fit)
  nm <- sub("^iv:", "lambda.", names(co))
  coefs <- as.list(unname(co))
  names(coefs) <- nm
  iterations <- fit$est.stat$nb.iter
  converged <- grepl("successive|gradient close|small", fit$est.stat$code, ignore.case = TRUE) ||
    fit$est.stat$code %in% c(1, 2)
  n_obs <- length(unique(df$case))
} else if (spec$fit == "logitr") {
  # its defaults otherwise: one start, its own Halton draws, the variance from
  # the Hessian
  suppressPackageStartupMessages(library(logitr))
  df <- df[order(df[[spec$panel]], df$case, df$alt), ]
  random <- unlist(spec$random)
  fit <- logitr(data = df, outcome = "chosen", obsID = "case", panelID = spec$panel,
                pars = spec$pars, randPars = random, numDraws = spec$draws,
                drawType = "halton", numCores = 1)
  co <- coef(fit)
  se <- sqrt(diag(vcov(fit)))
  nm <- sub("^sd_", "sd.", names(co))
  coefs <- as.list(unname(co))
  names(coefs) <- nm
  iterations <- fit$iterations
  converged <- fit$status > 0
  n_obs <- length(unique(df[[spec$panel]]))
  extra$loglik <- as.numeric(fit$logLik)
} else if (spec$fit == "alpaca") {
  suppressPackageStartupMessages(library(alpaca))
  fit0 <- alpaca::feglm(fml, data = df, family = binomial())
  fit <- biasCorr(fit0)
  se <- sqrt(diag(vcov(fit)))
  coefs <- as.list(coef(fit))
  extra$coefs_uncorrected <- as.list(coef(fit0))
  iterations <- fit0$iter
  converged <- isTRUE(fit0$conv)
  n_obs <- fit0$nobs[["nobs"]]
} else {
  stop("unknown fit ", spec$fit)
}
invisible(se)
t_fit <- now() - t0

cat("RESULT ", jsonlite::toJSON(c(list(
  t_read = t_read, t_fit = t_fit, n_obs = n_obs, iterations = iterations,
  converged = converged, peak_memory_mb = hwm_mb(), baseline_memory_mb = baseline,
  peak_disk_mb = 0, coefs = coefs), extra), auto_unbox = TRUE, digits = NA,
  na = "null"), "\n", sep = "")
