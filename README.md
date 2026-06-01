# Installation

## Clone Repository

git clone <repository_url>

cd credit-risk-model

## Create Virtual Environment

python -m venv venv

source venv/bin/activate

## Install Dependencies

pip install -r requirements.txt

# Data Source

Dataset:

Xente Fraud Detection Dataset

Source:
https://www.kaggle.com/competitions/xente-fraud-detection

The raw dataset should be placed in:

data/raw/data.csv

# Running EDA

jupyter notebook notebooks/eda.ipynb




Credit Scoring Business Understanding

# Credit Scoring Business Understanding

## 1. Basel II, Risk Measurement, and the Need for Interpretable Models

The Basel II Accord establishes international banking regulations that require financial institutions to maintain adequate capital reserves based on their risk exposure. A key principle of Basel II is that risk measurement models must be transparent, reliable, and well-documented so that regulators, auditors, and internal stakeholders can understand how risk assessments are produced.

This emphasis on risk measurement directly influences the design of credit scoring models. In a regulated financial environment, a model should not only generate accurate predictions but also provide explanations for its decisions. Financial institutions must be able to justify why a customer receives a particular risk score and demonstrate that the model's predictions are based on meaningful and measurable factors.

Consequently, interpretability and documentation become critical requirements. Every stage of the modeling process—including data preparation, feature engineering, model selection, validation, and monitoring—must be clearly documented. This ensures regulatory compliance, supports model governance, and enables ongoing validation and auditing of the credit risk framework.

## Basel II Internal Ratings-Based (IRB) Perspective

Under Basel II's Internal Ratings-Based (IRB) framework, banks may use internally developed models to estimate key risk parameters such as:

- Probability of Default (PD)
- Loss Given Default (LGD)
- Exposure at Default (EAD)

Although the Xente dataset does not contain actual default outcomes, the objective of this project closely aligns with the estimation of Probability of Default through alternative behavioral indicators.

The IRB framework places strong emphasis on:

1. Model transparency and explainability
2. Consistent documentation
3. Independent validation
4. Ongoing monitoring and recalibration
5. Demonstration of predictive power and stability

These principles influence the project architecture by motivating the use of reproducible pipelines, experiment tracking, documented feature engineering procedures, and explainable baseline models alongside more complex machine learning approaches.

## Model Governance and Validation

In regulated financial environments, model development extends beyond predictive accuracy.

A robust model governance framework includes:

- Version control of code and data transformations
- Documentation of assumptions and limitations
- Independent validation of model performance
- Monitoring for performance drift
- Periodic recalibration and retraining

To support these requirements, this project will utilize:

- GitHub for version control
- MLflow for experiment tracking and model versioning
- Automated testing for data processing pipelines
- Reproducible feature engineering workflows

## 2. The Need for a Proxy Variable and Associated Business Risks

The Xente transaction dataset does not contain a direct indicator of customer default. Since supervised machine learning algorithms require a target variable, it is necessary to construct a proxy variable that approximates credit risk based on observable customer behavior.

In this project, customer transaction behavior can be analyzed using Recency, Frequency, and Monetary (RFM) metrics. Customers who transact infrequently, spend relatively little, and exhibit long periods of inactivity may represent a higher-risk segment. By clustering customers according to their RFM profiles, a proxy target variable can be created to distinguish potentially high-risk customers from lower-risk customers.

While proxy variables make predictive modeling possible, they also introduce important business risks. The proxy does not represent actual loan default behavior and may therefore misclassify customers. Some customers identified as high risk may never default, while some customers labeled as low risk may eventually fail to repay their obligations. In addition, any assumptions or biases embedded in the proxy construction process may be propagated into the final model. As a result, predictions should be interpreted as estimates of behavioral risk rather than definitive measures of credit default probability.


## 3. Trade-Offs Between Interpretable and High-Performance Models

Credit risk modeling often requires balancing predictive performance with explainability.

### Logistic Regression with Weight of Evidence (WoE)

**Advantages**

* Highly interpretable and transparent.
* Model coefficients clearly indicate the direction and magnitude of risk factors.
* Widely accepted by regulators and risk management teams.
* Supports traditional credit scorecard development.
* Easier to validate, audit, and monitor.

**Disadvantages**

* Assumes primarily linear relationships between predictors and risk.
* May not capture complex interactions among variables.
* Often delivers lower predictive performance compared to advanced machine learning methods.

### Gradient Boosting Models

**Advantages**

* Typically achieve higher predictive accuracy.
* Capture nonlinear relationships and feature interactions automatically.
* Often perform well on complex behavioral datasets.

**Disadvantages**

* Less transparent and more difficult to explain.
* Individual predictions can be challenging to justify to regulators.
* Require additional tools and documentation to improve explainability.
* Increase model governance and monitoring complexity.

### Recommended Approach

In a regulated financial environment, model selection should consider both predictive performance and regulatory requirements. Logistic Regression with WoE provides strong interpretability and regulatory acceptance, while Gradient Boosting models may offer superior predictive power. A practical strategy is to evaluate both approaches and select the model that achieves an appropriate balance between accuracy, explainability, compliance, and business objectives.
