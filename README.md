# Marketing Campaign Analysis — Comparing Classifiers

Predicting which bank clients will subscribe to a term deposit, using data from a
Portuguese bank's telephone marketing campaigns (UCI Bank Marketing dataset,
~41,000 contacts). This project compares K-Nearest Neighbors, Logistic Regression,
Decision Trees, and Support Vector Machines.

**Full analysis:** [prompt_III.ipynb](prompt_III.ipynb)

## Business Objective

Every call an agent makes costs money, but only about **1 in 9 calls** results in a
subscription. The goal is to **focus calling effort on the clients most likely to
subscribe**, so the bank stops spending time and money on contacts that were never
going to convert.

## Data

- Source: [UCI Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
  (`data/bank-additional-full.csv`).
- 20 input features covering client demographics, the current/previous campaign, and
  social-economic context indicators.
- The `duration` feature is **excluded** from modeling: it is only known _after_ a
  call ends, so it cannot be used to decide who to call (target leakage).

## Approach

1. **EDA** — examined subscription rates across categorical features and distributions
   of continuous features (see the notebook's visualizations).
2. **Feature engineering** — handled the `pdays` 999 sentinel (split into a
   `was_previously_contacted` flag + a clean day count), kept `unknown` education as
   its own ordinal level, and one-hot / ordinal / scaled the remaining features inside
   a single `ColumnTransformer` pipeline.
3. **Modeling** — compared all four classifiers with default settings, then tuned each
   with `GridSearchCV` (5-fold cross-validation).
4. **Evaluation metric** — because the target is heavily imbalanced (~11% positive),
   models are scored on **F1 of the positive class** rather than accuracy, with
   `class_weight='balanced'` to keep the minority class from being ignored.

## Key Findings

- **Prior contact history is the strongest signal.** Clients contacted in a previous
  campaign subscribed ~63% of the time, versus ~9% for everyone else; a previous
  _successful_ outcome is especially predictive.
- **Economic context matters.** Indicators such as the Euribor rate, number of
  employees, and employment-variation rate correlate strongly with subscription.
- **Demographics are weak predictors.** Age, job, education, and marital status carry
  little signal on their own.
- **Accuracy is the wrong lens.** A "predict no for everyone" model is 88.9% accurate
  but finds zero subscribers. The tuned models reach ~84–86% accuracy but **catch
  roughly two-thirds of real subscribers** (recall ≈ 0.65).

## Recommendations

1. Prioritize previously-contacted clients, especially past "successes."
2. Time campaigns to favorable economic conditions.
3. Use the model to rank and shortlist leads instead of calling the full list.
4. Don't target on demographics alone.

## Next Steps

- Tune the decision threshold to real call economics (cost of a call vs. value of a
  subscription) to maximize profit rather than a generic F1 score.
- Run a cost-benefit analysis of the model shortlist vs. calling everyone.

## Repository Structure

```
.
├── prompt_III.ipynb        # Main analysis notebook
├── README.md               # This file
├── CRISP-DM-BANK.pdf       # Reference paper for the dataset
└── data/
    ├── bank-additional-full.csv
    ├── bank-additional.csv
    └── bank-additional-names.txt
```
