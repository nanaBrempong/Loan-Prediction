from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, QuantileDiscretizer, Bucketizer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

# Start Spark session
spark = SparkSession.builder.appName("LoanPrediction").getOrCreate()

# Load data
data = spark.read.csv('loan_data.csv', header=True, inferSchema=True)

# 1.Missing value handling
#  Handle missing values by dropping rows with any missing value for simplicity, there amount of missing data is small and this avoids complications of imputation
data = data.dropna()

# 2. Outlier treatment using IQR capping for main numeric columns
# We are keeping outliers within a reasonable range to prevent them from skewing the model, while still retaining important information that may be relevant for loan approval predictions
numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
for col_name in numeric_cols:
    quantiles = data.approxQuantile(col_name, [0.25, 0.75], 0.0)
    Q1, Q3 = quantiles
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data = data.withColumn(col_name, when(col(col_name) < lower, lower)
                                         .when(col(col_name) > upper, upper)
                                         .otherwise(col(col_name)))

# 3. Discretization of numeric columns into 4 bins
discretizers = [
    QuantileDiscretizer(numBuckets=4, inputCol='ApplicantIncome', outputCol='ApplicantIncome_bin'),
    QuantileDiscretizer(numBuckets=4, inputCol='CoapplicantIncome', outputCol='CoapplicantIncome_bin'),
    QuantileDiscretizer(numBuckets=4, inputCol='LoanAmount', outputCol='LoanAmount_bin')
]

for disc in discretizers:
    data = disc.fit(data).transform(data)

# Discretize Loan_Amount_Term
term_splits = [0, 120, 240, data.agg({"Loan_Amount_Term": "max"}).collect()[0][0]+1]
bucketizer = Bucketizer(splits=term_splits, inputCol="Loan_Amount_Term", outputCol="Loan_Amount_Term_bin")
data = bucketizer.setHandleInvalid("keep").transform(data)

# 4. Clean Dependents column
data = data.withColumn("Dependents", when(col("Dependents") == '3+', 3)
                                       .otherwise(col("Dependents").cast('int')))

# 5. Encode categorical variables
categorical_cols = ['Gender','Married','Education','Self_Employed','Property_Area',
                    'ApplicantIncome_bin','CoapplicantIncome_bin','LoanAmount_bin','Loan_Amount_Term_bin','Loan_Status']

indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx").fit(data) for c in categorical_cols]
for indexer in indexers:
    data = indexer.transform(data)

# 6. Feature selection
# We are dropping loan status as it is our target variable, and also  Loan_ID as it is not needed after encoding and may not provide additional predictive power
feature_cols = ['Gender_idx','Married_idx','Education_idx','Self_Employed_idx','Property_Area_idx',
                'ApplicantIncome_bin_idx','CoapplicantIncome_bin_idx','LoanAmount_bin_idx','Loan_Amount_Term_bin_idx','Dependents']

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# 7. Train/test split
train, test = data.randomSplit([0.8, 0.2], seed=42)

# 8. Train and evaluate classifiers
models = {
    "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="Loan_Status_idx"),
    "DecisionTree": DecisionTreeClassifier(featuresCol="features", labelCol="Loan_Status_idx"),
    "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="Loan_Status_idx")
}

evaluator = BinaryClassificationEvaluator(labelCol="Loan_Status_idx", metricName="areaUnderROC")

results = {}
for name, model in models.items():
    model_fit = model.fit(train)
    pred = model_fit.transform(test)
    auc = evaluator.evaluate(pred)
    results[name] = auc
    print(f"{name} AUC: {auc:.4f}")

# 9. Write results to output.txt
with open('output.txt', 'w') as f:
    for name, auc in results.items():
        f.write(f"{name} AUC: {auc:.4f}\n")
