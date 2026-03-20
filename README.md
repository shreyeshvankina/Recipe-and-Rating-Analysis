# NLP Recipe Rating Prediction

### Shreyesh Vankina & Ryan Rulkens

---

# Introduction

Online recipes often lack reliable feedback, making it difficult to know whether a recipe is worth trying. In this project, we analyze a large dataset of recipes to understand what influences recipe ratings and to build a model that predicts ratings.

Our central question is:

> **Do recipes with similar ingredients tend to receive similar ratings?**

The dataset contains **83,782 recipes** with features including:

* `minutes`: cooking time
* `n_steps`: number of steps
* `n_ingredients`: number of ingredients
* `ingredients`: ingredient list
* `tags`: recipe categories
* `nutrition`: nutritional values
* `average_rating`: average user rating

This question is important because it helps users evaluate recipes before cooking and improves recommendation systems.

---

# Data Cleaning and Exploratory Data Analysis

## Data Cleaning

We:

* Merged recipe and interaction datasets to compute average ratings
* Replaced invalid ratings (0) with missing values
* Extracted nutritional features (Calories, Sugar, Protein, etc.)
* Converted text columns (`tags`, `ingredients`) into usable formats
* Removed extreme outliers by trimming large values

Below is a preview of the cleaned dataset with some columns removed due to being complicated to display below:

| Index  | Name                                                        | Minutes | Steps | Ingredients | Avg Rating | Calories | Sugar |
|--------|------------------------------------------------------------ |--------:|------:|------------:|-----------:|---------:|------:|
| 26198  | dublin lawyer lobster dublin style with whiskey and cream   |      40 |     8 |           8 |          5 |    570.2 |     0 |
| 76699  | the peacock my version                                      |       3 |     5 |           6 |          5 |      104 |     0 |
| 59254  | poultry seasoning substitute                                |       1 |     1 |           2 |          5 |      2.6 |     0 |
| 44438  | lil bit of sunshine                                         |       3 |     4 |           3 |          5 |    138.7 |     0 |
| 56460  | peppermint chai tea                                         |      13 |     8 |           9 |          5 |      2.5 |     0 |
---

## Univariate Analysis

<iframe src="assets/ratings_hist.html" width="800" height="500"></iframe>

Most recipes have ratings between 4 and 5, indicating strong positive bias in user reviews.

---

## Bivariate Analysis

<iframe src="assets/steps_vs_rating.html" width="800" height="500"></iframe>

There is little relationship between number of steps and rating, suggesting complexity does not strongly affect user satisfaction.

---

## Aggregated Analysis

We grouped recipes by number of ingredients and steps and computed average ratings.

This showed that ratings remain relatively stable across different recipe complexities.

Below is small part of pivot table:

| n_ingredients | 5 steps | 10 steps | 15 steps |
|-------------|---------|----------|----------|
| 3 | 4.67 | 4.55 | 4.60 |
| 5 | 4.70 | 4.62 | 4.58 |
| 8 | 4.72 | 4.65 | 4.61 |

---

# Assessment of Missingness

## MNAR Analysis

The `average_rating` column is likely **Missing Not At Random (MNAR)**.

Users are more likely to rate recipes they enjoyed, meaning missing ratings may correspond to worse recipes.

Additional user behavior data could help explain this missingness.

---

## Missingness Dependency

Permutation tests show:

* Missingness depends on:

  * `minutes` (p ≈ 0.0)
  * `n_steps` (p ≈ 0.0)
* Weak dependence on:

  * `n_ingredients` (p ≈ 0.03)

<iframe src="assets/missingness_minutes.html" width="800" height="500"></iframe>

<iframe src="assets/missingness_step_number.html" width="800" height="500"></iframe>

<iframe src="assets/missingness_ingredient_number.html" width="800" height="500"></iframe>

Longer recipes are less likely to receive ratings.

---

# Hypothesis Testing

## Hypotheses

* **Null:** Ingredient similarity is not associated with rating similarity
* **Alternative:** Recipes with similar ingredients have similar rating

## Results

* Observed statistic: -0.063
* p-value: 0.237
* Significance level: α = 0.05

We did a permutation test because we want to check whether ingredient similarity
and rating similarity are related without assuming any specific distribution.

Our test statistic measures the difference in ratings between recipes that have
similar ingredients and recipes that do not. This directly matches our question,
which asks whether recipes with similar ingredients tend to receive similar ratings.

A permutation test is a good choice because it lets us simulate what would happen
if ingredient similarity and rating similarity were unrelated, and compare our
observed result to that scenario.

## Conclusion

We fail to reject the null hypothesis.

There is not enough evidence that recipes with similar ingredients receive similar ratings.

---

# Framing a Prediction Problem

We aim to predict the average rating of a recipe.

* **Type:** Regression  
* **Response Variable:** Mean recipe rating  

We chose rating as the response variable because ratings summarize how much
users liked a recipe, and predicting ratings could help estimate the quality
of recipes that have not yet been reviewed.

* **Features:**
  * Numerical: minutes, number of steps, nutrition values
  * Text: ingredients, tags

These features are known at the time a recipe is created, so they would be
available when making a prediction. We do not use user ratings as features,
since ratings are only known after users review the recipe.

### Metric

We use **R²** and **Root Mean Squared Error (RMSE)** to evaluate the model.

R² measures how much variation in ratings the model explains, while RMSE
measures how close the predicted ratings are to the true ratings. These
metrics are appropriate for regression problems where the response variable
is continuous.

---

# Baseline Model

We built a baseline regression model to predict the average rating.

### Features used

- `minutes` (quantitative)
- `n_steps` (quantitative)
- `n_ingredients` (quantitative)
- `Sugar` (quantitative)

### Transformations

- Log transform on `minutes` and `Sugar` because scatter plot showed a log relationship
- Polynomial features on `n_steps` and `n_ingredients` because scatter plot showed a polynomial relationship

These transformations were used to allow the model to capture
non-linear relationships between recipe complexity and rating.

### Model

We used **Linear Regression** inside a sklearn Pipeline.

### Performance

- R² ≈ 0.002  
- RMSE ≈ 0.63  

The performance is very low, meaning the baseline model does not
explain much variation in ratings. This suggests that simple numeric
features alone are not enough to predict ratings well.

---

# Step 7: Final Model

For our final model, we decided to use a RandomForestRegressor and experimented with different hyperparameters and features. However, it was very hard to improve models to predict the continuous column `average_rating` so we decided to move on toward a different approach. We transformed the problem into a classification task by rounding ratings to the nearest integer, allowing the model to focus on broader rating categories.

## Features Added

- **`steps_per_minute`** – Captures the pace of a recipe by dividing the number of steps by total cooking time. Recipes with many steps but short times may be more complex, affecting ratings.
- **Polynomial features of numeric variables** – Includes interactions and squared terms for `minutes`, `n_steps`, `n_ingredients`, `Calories`, and `Protein` to capture non-linear relationships with ratings.
- **Text vectorization** – Vectorized `tags` and `ingredients` using `CountVectorizer` to see whether certain tag and ingredient patterns can capture people's attention and have higher ratings.
- **`Sugar` binarization** – having recipes with high sugar may have higher ratings since most people might like sweet food


## Modeling Approach

We tested multiple classification algorithms such as **Logistic Regression** and **Random Forest Classifier**, using a `Pipeline` to combine preprocessing and modeling. Hyperparameter and feature transformation combinations were selected using `GridSearchCV` with 3-fold cross-validation:

- **Logistic Regression:** tested `C` and solver.
- **Random Forest:** tested `n_estimators`, `max_depth`, `min_samples_split`, and `max_features`.

### Best Model Hyperparameters (LogisticRegression(max_iter=1000))
- `model__C: 0.1`
- `model__solver: lbfgs`

This model had the highest accuracy while also accounting for overfitting and computer cost.

## Model Performance

- Train Accuracy ≈ 0.69  
- Test Accuracy ≈ 0.69  

The model worked better because we grouped ratings into categories and added details about how complex a recipe is and what's in it.
