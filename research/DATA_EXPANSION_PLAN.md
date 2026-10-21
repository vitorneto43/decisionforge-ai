# DecisionForge — Real-data expansion plan

Status: **data-source research only**. Listing a dataset does not mean our models have ingested or learned it.

## 1. Banking prediction (current)
- UCI Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing
- 41,188 chronologically ordered rows in bank-additional-full.csv, May 2008–Nov 2010.
- Target is the observed subscription after contact, **not** the causal effect of deciding to call.
- Keep call duration out of pre-call predictors. Avoid using future information and overlapping historical samples as independent test sets.
- Existing historical test period was already inspected during V1 work; treat it as development evidence, NOT a fresh final holdout.

## 2. Business operations (new independent problem)
- UCI Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- ~1,067,371 transactions, Dec 2009–Dec 2011, from a UK online retailer.
- License: CC BY 4.0; credit the source and original author (Daqing Chen).
- Possible task: forecast next-period demand by stock code using only prior sales, aggregated into weekly buckets. A sale is not a decision outcome.
- Handle returns/cancellations, missing customer IDs, outliers, item cold-starts and class/time imbalance.
- Split by time and prevent mixing the same target period into earlier features.

## Rules before training
1. Obtain data from official source and record download URL, citation, license, retrieval date and SHA-256.
2. Validate schema, chronological availability, missingness and duplicates.
3. Name a specific outcome and **what is known at prediction time**.
4. Compare against strong trivial/statistical baselines; report both aggregate and subgroup metrics.
5. Lock a genuinely unseen later period or external source; no tuning against its scores.
6. Calibrate uncertainty and report coverage versus error; an observed 97% accuracy on selected predictions is not a cross-domain safety certificate.
7. Do not ship a high-stakes decision service based on these public observational datasets.

## Constraints
- No external pretrained AI models or commercial AI API calls.
- NumPy and ordinary file-processing libraries are permissible infrastructure.
- Do not add raw datasets to the git repository; download into ignored local data paths.
- Data quantity alone does not guarantee generalization, especially across different tasks.
- Banking and retail use different outcome labels and schemas; do not simply append rows into one training array.

## Progress checklist
- [x] Banking dataset investigated and used for exploratory supervised evaluation.
- [x] Business retail dataset identified.
- [ ] Download and validate the retail source.
- [ ] Implement causal-time-safe retail features and naive forecasting baselines.
- [ ] Train an independently evaluated neural retail forecasting model.
- [ ] Obtain fully external chronological banking holdout.
- [ ] Define domain-specific acceptance criteria and assess generalization.
