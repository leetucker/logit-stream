"""Build the benchmark data from the ACS Public Use Microdata Sample.

    python benchmarks/prepare_acs.py            # 2019-2023 5-year file
    python benchmarks/prepare_acs.py --one-year # 2023 1-year file

Downloads (if not already in benchmarks/data/raw/, or $LOGIT_BENCH_CACHE) the person records of the
American Community Survey PUMS from the Census Bureau

    https://www2.census.gov/programs-surveys/acs/data/pums/2023/5-Year/csv_pus.zip

(2.2 GB zipped, 10 GB of CSV, 16 million people) and writes the adults (18
and over) as benchmarks/data/acs_2023_5yr.parquet, in a random order, so that
the first n rows are a simple random sample of n adults.

The outcomes are those of a demographic imputation:

    hispanic   0/1: Hispanic or Latino origin (HISP != 1)
    race       six categories in the style of BISG: Hispanic (any race),
               and among the rest White, Black, Asian or Pacific Islander,
               American Indian or Alaska Native, and other or multiracial
    race5      the same with AIAN merged into other (0.8% of adults: rare
               enough to be absent from some divisions in small samples)
    educ       five ordered levels of attainment for those 25 and over
               (null for 18-24): less than high school, high school or
               GED, some college or associate's, bachelor's, graduate

and the covariates characteristics an imputation might use: age group,
sex, personal income group (in 2023 dollars), nativity, citizenship (born
in a territory counted with the US-born), English ability ("not well" and
"not at all" merged), marital status, disability, Census division, state and
PUMA (``puma_id``, unique across states).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
# the download; the EC2 runner keeps it on the root volume, which survives a
# spot stop (the data directory, on an instance store, does not)
RAW = Path(os.environ.get("LOGIT_BENCH_CACHE", HERE / "data" / "raw"))
URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/{}/csv_pus.zip"

COLUMNS = ["SERIALNO", "SPORDER", "DIVISION", "STATE", "PUMA", "PWGTP", "ADJINC", "AGEP", "SEX",
           "RAC1P", "HISP", "SCHL", "PINCP", "NATIVITY", "CIT", "ENG", "MAR", "DIS"]


def build(lf: pl.LazyFrame) -> pl.LazyFrame:
    age = pl.col("AGEP")
    inc = pl.col("PINCP") * pl.col("ADJINC") / 1e6
    hisp = pl.col("HISP") != 1
    r = pl.col("RAC1P")
    race = (pl.when(hisp).then(pl.lit("Hispanic"))
              .when(r == 1).then(pl.lit("White"))
              .when(r == 2).then(pl.lit("Black"))
              .when(r.is_in([6, 7])).then(pl.lit("API"))
              .when(r.is_in([3, 4, 5])).then(pl.lit("AIAN"))
              .otherwise(pl.lit("Other")))
    s = pl.col("SCHL")
    educ = (pl.when(age < 25).then(None)
              .when(s <= 15).then(0).when(s <= 17).then(1).when(s <= 20).then(2)
              .when(s == 21).then(3).otherwise(4))
    agecat = (pl.when(age < 25).then(0).when(age < 35).then(1).when(age < 45).then(2)
                .when(age < 55).then(3).when(age < 65).then(4).when(age < 75).then(5)
                .otherwise(6))
    inccat = (pl.when(inc <= 0).then(0).when(inc < 10_000).then(1).when(inc < 25_000).then(2)
                .when(inc < 50_000).then(3).when(inc < 75_000).then(4)
                .when(inc < 100_000).then(5).otherwise(6))
    return (lf.select(COLUMNS)
              .filter(age >= 18)
              .select(
                  pl.col("SERIALNO").cast(pl.Utf8).alias("serialno"),
                  pl.col("DIVISION").cast(pl.Int8).alias("division"),
                  pl.col("STATE").cast(pl.Int16).alias("state"),
                  (pl.col("STATE").cast(pl.Int32) * 100_000 + pl.col("PUMA").cast(pl.Int32))
                  .alias("puma_id"),
                  pl.col("PWGTP").cast(pl.Float64).alias("weight"),
                  age.cast(pl.Int16).alias("age"),
                  agecat.cast(pl.Int8).alias("agecat"),
                  (pl.col("SEX") == 2).cast(pl.Int8).alias("female"),
                  inccat.cast(pl.Int8).alias("inccat"),
                  (pl.col("NATIVITY") == 2).cast(pl.Int8).alias("foreign_born"),
                  # born in Puerto Rico or another territory (CIT 2) counts with the
                  # US-born, and "not well" with "not at all" (ENG 3, 4): both
                  # are rare enough that a small sample can have no one of some
                  # race in them, which separates their coefficients
                  pl.col("CIT").replace(2, 1).cast(pl.Int8).alias("cit"),
                  pl.col("ENG").fill_null(0).replace(4, 3).cast(pl.Int8).alias("eng"),
                  pl.col("MAR").cast(pl.Int8).alias("mar"),
                  (pl.col("DIS") == 1).cast(pl.Int8).alias("disabled"),
                  hisp.cast(pl.Int8).alias("hispanic"),
                  race.alias("race"),
                  pl.when(race == "AIAN").then(pl.lit("Other")).otherwise(race).alias("race5"),
                  educ.cast(pl.Int8).alias("educ"),
              ))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-year", action="store_true")
    args = ap.parse_args(argv)
    kind = "1-Year" if args.one_year else "5-Year"
    tag = "1yr" if args.one_year else "5yr"
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / f"csv_pus_2023_{tag}.zip"
    if not zpath.exists():
        print(f"downloading {URL.format(kind)}", flush=True)
        part = zpath.with_suffix(".part")
        subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", str(part), URL.format(kind)],
                       check=True)
        part.rename(zpath)
    out = HERE / "data" / f"acs_2023_{tag}.parquet"
    tmp = HERE / "data" / f"csv_{tag}"          # 10 GB of CSV, beside the data
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        for n in names:
            if not (tmp / n).exists():
                print(f"extracting {n}", flush=True)
                z.extract(n, tmp)
    parts = []
    for n in names:
        lf = pl.scan_csv(tmp / n, schema_overrides={"SERIALNO": pl.Utf8, "PUMA": pl.Utf8,
                                                    "STATE": pl.Utf8, "DIVISION": pl.Utf8},
                         infer_schema_length=10_000)
        part = tmp / (n[:-4] + ".parquet")
        build(lf).sink_parquet(part)
        parts.append(part)
        print(f"  {n} -> {part.name} ({time.time() - t0:.0f} s)", flush=True)
    frame = pl.concat([pl.read_parquet(p) for p in parts])
    frame = frame.sample(fraction=1.0, shuffle=True, seed=20261027)
    frame.write_parquet(out, row_group_size=250_000)
    print(f"{frame.height:,} adults -> {out} ({time.time() - t0:.0f} s)")
    print(frame.select(pl.col("race").value_counts(normalize=True)).unnest("race"))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
