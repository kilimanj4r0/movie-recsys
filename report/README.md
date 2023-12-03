# Final Report

## Introduction

The goal of the recommender system is to suggest some movies to the user based on users features and favorite movies. We can approach this problem as a top-`k` recommendation task, where model will recommend `k` movies for the user and recommendations will be sorted by score. Since we have data about time (timestamps in seconds) in our interactions dataset (`u.data`) we can use *collaborative filtering* (presented below). As a subdirection, I choosed model-based approach.

![Collaborative filtering](https://miro.medium.com/v2/resize:fit:4800/format:webp/0*2huiFTsBStKKkaWb.png)

In this assignment, I used [recently released by MTS the RecTools framework](https://habr.com/ru/articles/773126/) to build a solution.

![RecTools escription](https://habrastorage.org/r/w1560/getpro/habr/upload_files/1b7/2c3/e52/1b72c3e52e5ec0a878cda42098bac20a.png)

## Data analysis

First, splits of the data given (`u1-u5`, `ua`, `ub`) are divided by date, so that we have past interactions in the train set and current iterations in the test set.
Second, I made the Exploratory Data Analysis ([notebook](/notebooks/1.0-eda.ipynb)) and found the following insights:
- Average rating score is 3.52
- Top 1 genre is Comedy (30% of movies)
- Least popular genre is Fantasy (1.3% of movies)
- Average user age is 34 years old (ages from 7 to 73 years old are in the dataset)
- Ratio of user gender is 71% / 29% for Male / Female
- Student is the most popular occupation (20.7% of users)
- Homemaker and Doctor are least popular occupations (0.74% of users)
- For 943 users there are 795 unique zip codes

These analysis results are used in the next step, data preprocessing.

## Data preprocessing

Considerable amount of time were spent on this stage ([notebook](/notebooks/1.1-data-preprocessing.ipynb)). I converted the given data into RecTools [dataset format](https://rectools.readthedocs.io/en/stable/api/rectools.dataset.dataset.Dataset.html#rectools.dataset.dataset.Dataset). I also used `pandas` library on this stage to work with dataframes. The given `u1-u5` data splits were used for cross-validation, so I stored files into `data/interim/` directory. For the test data, I decided to use `ua` and `ub` splits and stored them into `benchmark/data/` folder to access them on the evaluation stage. The processing was the following:

### User features

- Dropped `zip_code` column
- Normalized `age` column (divide each value by maximum `age` present)
- One-hot encoded `gender` and `occupation` (squeezing)
- Created as sparse

### Item features

- Dropped `title`, `release_date`, `video_release_date`, `IMDB_URL` columns
- Created as dense

### Data splits

- For features, only those ids were kept that are present in the `base` split
- Casted `timestamps` into `datetime` format
- Casted `rating` into `float` and treated as `Weight`


As a result, I saved both RecTools datasets and dataframes to load them for training and evaluation stages.

## Model Implementation

I implemented 5-fold cross-validation on the `u1-u5` dataset splits to choose the best model ([notebook](/notebooks/2.1-train-cross-val.ipynb)). The choosing criteria was Root Mean Squared Error (RMSE) between predicted and test rating score. I have used implementation of the following models in RecTools:
- `PureSVDModel` (utilizes Singular Value Decomposition for matrix factorization to recommend top-k items)
- `ImplicitALSWrapperModel` (the wrapper of `AlternatingLeastSquares` collaborative filtering model using implicit feedback, recommending top-k items) with `10` sized factors vector for `K=10`
- `LightFMWrapperModel` (the wrapper of `LightFM` hybrid model combining collaborative and content-based signals to recommend top-k items) with `10` sized factors vector for `K=10`

Averaging RMSE across all folds for each model, give the following results:
|Model|RMSE|
|-|-|
|`PureSVDModel`|2.188761440434592|
|`ImplicitALSWrapperModel`|2.4058378050082134|
|`LightFMWrapperModel`|29.475618221101787|

So, attempt to use hybrid model was unsuccessful. `AlternatingLeastSquares` was really close to `PureSVDModel` but still worse. Although `AlternatingLeastSquares` are more used in practice, I choosed `PureSVDModel` as the best one. 

## Model Advantages and Disadvantages

The `PureSVDModel` has the following advantages and disadvantages:

### Pros:

- Simple and clear
- Straightforward due to singular value decomposition (SVD)
- Handles Cold Start problem
- Performs good for dense data

### Cons:
- Interpretable reommendations
- Limited to Matrix Factorization
- Treats all users and items equally
- Have Cold Start problem for new items:
- Sensitive to sparse data:

## Training Process

On this stage, I used training process suggested by RecTools ([notebook](/notebooks/3.0-evaluation.ipynb)). First, the `PureSVDModel` model was fitted (`fit()` method) on the train (`base` in our case) split of the dataset. Next, to build recommendations table for `k=10` recommendations per used the `recommend()` method was used. After, for cross-validation the RMSE was computed between predicted and test rating score, while for evaluation the recommendations metrics were computed (described in the next step).

## Evaluation

I computed Classification and Ranking metrics [implemeted in RecTools](https://rectools.readthedocs.io/en/stable/api/rectools.metrics.html), namely:
- F1Beta — F-score for k first recommendations
- Mean Average Precision (MAP) — takes mean value of precision of recommendations taking into account their order
- Mean Reciprocal Rank (MRR) — takes mean value of relevance of recommendations taking in account their order.

By the inspiration of [medium article](https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff) I also added for the interest these metrics:
- Mean Inverse User Frequency (Novelty) — How surprising are the recommendations in general?
- Serendipity (Novelty + Relevance) — How surprising are the relevant recommendations?

For all metrics, diffrent $k=[1, 5, 10]$ were used ([notebook](/notebooks/3.0-evaluation.ipynb)). The choice of metrics also inspired by [this article](https://neptune.ai/blog/recommender-systems-metrics) and [this notebook](https://github.com/MobileTeleSystems/RecTools/blob/main/examples/5_benchmark_iALS_with_features.ipynb).

## Results

The `PureSVDModel` evaluation results on the data splits `ua` and `ub` are presented here:

| Metric           | Fold `a`                 | Fold `b`                 | Average  |
|------------------|------------------------|------------------------|------------------------|
| F1Beta@1         | 0.0825                 | 0.0812                 | 0.0825                 |
| F1Beta@5         | 0.2054                 | 0.1996                 | 0.2054                 |
| F1Beta@10        | 0.2372                 | 0.2316                 | 0.2372                 |
| MRR@1            | 0.4539                 | 0.4464                 | 0.4539                 |
| MRR@5            | 0.5888                 | 0.5773                 | 0.5888                 |
| MRR@10           | 0.6030                 | 0.5927                 | 0.6030                 |
| MAP@1            | 0.0454                 | 0.0446                 | 0.0454                 |
| MAP@5            | 0.1134                 | 0.1106                 | 0.1134                 |
| MAP@10           | 0.1457                 | 0.1420                 | 0.1457                 |
| Novelty@1        | 1.6174                 | 1.6123                 | 1.6174                 |
| Novelty@5        | 1.8300                 | 1.8311                 | 1.8300                 |
| Novelty@10       | 1.9767                 | 1.9755                 | 1.9767                 |
| Serendipity@1    | 0.0066                 | 0.0064                 | 0.0066                 |
| Serendipity@5    | 0.0065                 | 0.0065                 | 0.0065                 |
| Serendipity@10   | 0.0060                 | 0.0060                 | 0.0060                 |

I also averaged the results (last column) to get the whole picture.

