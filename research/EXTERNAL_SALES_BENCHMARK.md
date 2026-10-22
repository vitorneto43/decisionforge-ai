# External retailer benchmark

Source: James Tan, **Sales Transactions Weekly**, UCI Machine Learning Repository (CC BY 4.0). https://doi.org/10.24432/C5XS4Q. Dataset: 811 products, 52 weeks.

Download from https://archive.ics.uci.edu/dataset/396/sales+transactions+weekly and save `Sales_Transactions_Dataset_Weekly.csv` or its official ZIP under `data/`.

Run:
```bash
python research/external_sales_benchmark.py --csv-or-zip data/Sales_Transactions_Dataset_Weekly.csv
python -m unittest discover -s tests -v
```

This experiment trains a **new** NumPy MLP from random initial weights on the new dataset; it tests *method replication*, NOT transfer of pretrained weights from Online Retail II. Only W0..W51 are used, excluding normalization and min/max values calculated from future sales. Training targets weeks 8-31; validation 32-41; test 42-51. Report MAE and RMSE against mean of previous four weeks.

No measured results on the actual second dataset yet. This does not establish 97% decision reliability, causal effects, or safe business advice. Do not tune to the test scores repeatedly.
