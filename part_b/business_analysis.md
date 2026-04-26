B1. Problem Formulation

B1(a) — ML Problem Formulation

Target Variable: Number of items sold per store per month (sales volume), under a specific promotion.

Input Features:
- Store features: location type (urban/semi-urban/rural), store size, monthly footfall, local competition density
- Customer demographics: age group, income level, etc.
- Promotion type: Flat Discount, BOGO, Free Gift, Category-Specific, Loyalty Points
- Calendar features: month, weekend count, festival flags

Type of ML Problem: This is a supervised multiclass classification problem — given a store and month, predict which promotion (out of 5) will result in the most items sold.

Alternatively, it can be framed as a regression problem where you predict expected items sold for each promotion and then pick the highest. This is arguably better because it gives a concrete number, not just a label.

Justification: We have historical data with known outcomes (which promotion ran, how many items were sold), so supervised learning fits. Since we're choosing from 5 discrete promotions, classification (or regression-then-rank) is the right approach.



 B1(b) — Why Items Sold Over Revenue?

Revenue is affected by price, discounts, and promotion costs — things that vary independently of how well the promotion actually worked. For example, a Flat Discount might show lower revenue but push far more units, which is what we actually want to know.

Items sold is a cleaner, more direct measure of promotion effectiveness. It's not distorted by pricing changes.

Broader Principle: In ML, the target variable should directly measure what the business is actually trying to optimise — not a proxy that looks related but introduces noise. Choosing the wrong target leads to a model that's technically accurate but answers the wrong question.



 B1(c) — Against a Single Global Model

A single model assumes all 50 stores behave similarly. But a rural store and a city-centre store will respond very differently to the same promotion — different customer needs, competition levels, and footfall patterns.

Better Strategy: Cluster-based or Hierarchical Modelling

Group stores by type (e.g., urban, semi-urban, rural — or by size and footfall) and train a separate model per group. This way, each model learns the patterns relevant to stores it actually applies to.

An even better approach is a mixed-effects model or adding store-level features (like location type and size) as inputs, which lets the model personalise recommendations without needing 50 completely separate models.




 B2. Data and EDA Strategy

 B2(a) — Joining Tables and Dataset Grain

How to join:
1. Start with the transactions table (one row per transaction).
2. Join store attributes on `store_id` to bring in location type, size, footfall, etc.
3. Join promotion details on `promotion_id` to get the promotion type active during that period.
4. Join the calendar table on `date` to add weekend flags, festival indicators, and month/year.

Grain of the final dataset: One row = one store, one month, one promotion. This means you aggregate transactions up to the store-month level before modelling.

Aggregations to perform:
- Total items sold per store per month (target variable)
- Total transactions per store per month
- Average basket size
- Footfall (if tracked daily, sum or average to monthly)



 B2(b) — EDA to Perform

1. Promotion Performance by Type
Bar chart of average items sold per promotion type. This shows if some promotions are generally stronger and helps set expectations before modelling.

2. Performance by Store Location
Box plots of items sold grouped by urban/semi-urban/rural. If rural stores respond very differently from urban ones, this confirms we need location as a key feature and possibly separate models.

3. Time Trends and Seasonality
Line chart of monthly items sold over 3 years. Look for seasonal spikes (festivals, holidays). This tells us whether month/season should be a feature and if time-based splitting is needed.

4. Promotion-Location Interaction
Heatmap of average items sold by promotion type × store location. If BOGO works well in urban stores but poorly in rural ones, we know the model needs to learn this interaction — and we might engineer an interaction feature.

5. Correlation Analysis
Check how footfall, store size, and competition density correlate with items sold. Drop features that show near-zero correlation or are highly correlated with each other (multicollinearity).



 B2(c) — Handling Class Imbalance (80% No-Promotion)

If most data has no promotion, the model will learn that "no promotion" is the norm and will be biased towards predicting outcomes as if no promotion is running. It won't learn well how promotions actually differ from each other.

Steps to address this:

- Filter or separate: If the task is purely about choosing which promotion to run (not whether to run one), filter the dataset to only include rows where a promotion was active. The no-promotion data can be kept as a baseline for comparison but shouldn't dominate training.
- Resampling: Oversample under-represented promotion types using SMOTE or simple duplication, or undersample the no-promotion records.
- Class weights: If using a classifier, assign higher weights to promotion classes so the model penalises getting those wrong more heavily.




 B3. Model Evaluation and Deployment

 B3(a) — Train-Test Split and Metrics

How to split:
Use a time-based split, not random. For example:
- Training: Month 1 to Month 30 (2.5 years)
- Testing: Month 31 to Month 36 (last 6 months)

Why random split is wrong:
Random splitting allows future data to leak into training (e.g., the model trains on December 2024 and tests on March 2024). This makes performance look artificially good. In reality, the model always predicts the future using the past — the split must reflect that.

Evaluation Metrics:

1-RMSE(if regression): Average prediction error in number of items. Lower is better. 
2-MAE: Less sensitive to big errors. Easier to explain to stakeholders. 
3-Accuracy / F1 (if classification): What % of promotion choices were correct. F1 handles imbalance better. 
4-Lift / Rank correlation  Did stores that got the recommended promotion actually sell more? Business-level check. 

The most important business check is when we follow the model's recommendation, do stores sell more items than when they don't.



 B3(b) — Feature Importance to Explain Recommendations

Feature importance tells you which inputs the model relied on most when making a prediction. We can get this from tree-based models directly, or use SHAP values for any model.

For Store 12:
- In December, the model likely sees high festival flags, high footfall, and a customer base that responds to loyalty rewards. SHAP values would show "festival = yes" and "footfall = high" pushing strongly towards Loyalty Points Bonus.
- In March, there's no festival, footfall is lower, and competition might be higher — SHAP values would show "month = March" and "competition density = high" pushing towards Flat Discount (a straightforward price incentive).

How to communicate this to the marketing team:
Show a simple bar chart of SHAP values for Store 12 for each month — no jargon. Label the bars with plain English: "The model recommends Loyalty Points in December mainly because it's festival season and footfall is at its peak." This makes the model's logic transparent and builds trust.



 B3(c) — End-to-End Deployment

Step 1 — Save the model:
After training, save the model using a standard format like `pickle` or `joblib` (for Python sklearn/XGBoost models). Also save the preprocessing pipeline (scalers, encoders) separately so new data can be transformed the same way.

Step 2 — Prepare new monthly data:
At the start of each month, run an automated pipeline that:
- Pulls last month's transactions, footfall, and store data from the database
- Joins with the promotion calendar and festival flags for the upcoming month
- Applies the same feature engineering steps used during training
- Feeds the processed data into the saved model

Step 3 — Generate and serve recommendations:
The model outputs a recommended promotion for each of the 50 stores. These go into a dashboard or directly to the marketing team's planning tool.

Step 4 — Monitoring (detecting model degradation):

- Prediction drift: Track whether the distribution of recommended promotions changes significantly month over month (e.g., the model suddenly recommends Loyalty Points for 45 out of 50 stores every month — that's a red flag).
- Outcome tracking: After each month ends, compare predicted items sold vs. actual items sold. Track RMSE or MAE over time. If error starts rising consistently, the model is degrading.
- Data drift: Monitor if input features (footfall, demographics, competition) are shifting in ways the model hasn't seen. Use statistical tests (e.g., KS test) to detect this automatically.

When to retrain: Set a threshold — for example, if MAE rises more than 15% above the baseline for two consecutive months, trigger a retraining job using the latest available data.