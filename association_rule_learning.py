############################################
# ASSOCIATION RULE LEARNING
############################################

"""
Our aim is to suggest products to users in the product
purchasing process by applying association analysis to
the online retail II dataset.
"""

# 1. Data Preprocessing
# 2. Preparing ARL Data Structure (Invoice-Product Matrix)
# 3. Extracting Association Rules
# 4. Functionalization of Work
# 5. Recommending Products to Users at Cart Stage

############################################
# Data Preprocessing
############################################

!pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None) # show all columns
pd.set_option('display.width', 500) # show 500 side by side
pd.set_option('display.expand_frame_repr', False) # ensures that the output is on a single line
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("datasets/csv_path/w3/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()

def check_df(dataframe, head = 5):
    print("######### Shape #########")
    print(dataframe.shape)
    print("######### Types #########")
    print(dataframe.dtypes)
    print("######### Head #########")
    print(dataframe.head(head))
    print("######### Tail #########")
    print(dataframe.tail(head))
    print("######### NA #########")
    print(dataframe.isnull().sum())
    print("######### Quantiles #########")
    print(dataframe.quantile([0,0.05,0.05,0.05,0.99,1]).T)

"""
A function to calculate threshold values.
Values outside the general distribution of a variable are called outliers.
How to catch outliers?
-> If we check the measurements of the variables themselves, outliers are caught for the relevant ones.
"""

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Function that will replace outliers with threshold values according to the threshold values.
# If a value is greater than the upper limit, replace the value of this variable with up_limit.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable ] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace = True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na = False)] # delete those with C at the beginning
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity") # The variable Quantity replaces the existing outliers with the limit values of these variables.
    replace_with_thresholds(dataframe, "Price") # Replace the existing outliers of the price variable with the limit values of these variables.
    return dataframe
df = retail_data_prep(df)

# If we don't want to call recursive functions
# Create a file called Helpers
# We can call functions from that directory
# from helpers.helpers import retail_data_prep, check_df

############################################
# Preparing ARL Data Structure (Invoice-Product Matrix)
############################################

# Write 1 if products exist, 0 otherwise.

df_ge = df[df['Country'] == "Germany"]
check_df(df_ge)

# Indicates how many of each product was purchased.
df_ge.groupby(['Invoice','Description']).agg({"Quantity":"sum"}).head(20)

"""
Required:
Have only one invoice code per line
and the names of the variables (products) in the columns.

Intersection:
In intersections, only information on whether there is a product in those invoices

-> Not concerned with how much
-> It is concerned with whether or not it exists.
-> In other words, the table should be pivoted.
-> The expressions in the description are requested to pass into the column.
"""

# Rows have cart(Invoice), columns have products (Description).
# If a product is in a cart, how many information does it have.
df_ge.groupby(['Invoice','Description']).agg({"Quantity":"sum"}).unstack().iloc[0:5,0:5]

# Required: lines must be written either present or absent (0 or 1)
# NaN = 0 must be
# NaN is filled with zero.
df_ge.groupby(['Invoice','Description']).agg({"Quantity":"sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# With "\" we can split the codes into sub-lines.
# Write 1 if a value is greater than zero, otherwise write 0.

"""
apply: apply to row or rows
applymap: It has the ability to apply a dataframe to the entire function, not by row or column.
Task: Look at all the observations, if there are values greater than zero, write a text, otherwise write zero
"""
df_fr.groupby(['Invoice','Description']).\
    agg({"Quantity":"sum"}).\
    unstack().\
    fillna(0).\
    applymap(lambda x: 1 if x>0 else 0).iloc[0:5,0:5]

def create_invoice_product_df(dataframe, id = False):
    if id:
        return dataframe.groupby(['Invoice','StockCode'])['Quantity'].sum().unstack().fillna(0).\
            applymap(lambda x: 1 if x>0 else 0)
    else:
        return dataframe.groupby(['Invoice','Description'])['Quantity'].sum().unstack().fillna(0).\
            applymap(lambda x: 1 if x>0 else 0)

# Do according to StockCode if id is entered
# If id is false, do it according to the description

ge_inv_pro_df = create_invoice_product_df(df_ge)
ge_inv_pro_df.head()

ge_inv_pro_df = create_invoice_product_df(df_ge, id=True)
ge_inv_pro_df.head()

# which id is on which product
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)
check_id(df_ge, 23084)

############################################
# Association Rule
############################################

# The appriori function is used for the probabilities of all possible product combinations.
freaquent_itemsets = apriori(ge_inv_pro_df, min_support = 0.01, use_colnames = True)

# min_support = thresholds value that we set.
# use_colnames = use column names-> product ids will come

# Probability of products appearing alone
# sort by support value in descending order
# support =0.245077 -> probability of being observed in all data is 24%.
freaquent_itemsets.sort_values("support", ascending = False).head(20)

rules = association_rules(freaquent_itemsets, metric = "support", min_threshold = 0.01)
rules.sort_values("support", ascending = False).head()

"""
Calculated only support with apriori
All other metrics calculated with association rules
"""

"""
- antecedents = first product
- consequents = second product
- antecedent support = probability of first product
- consequent support = probability of second product
- support = the probability of two products appearing together
- confidence = The probability of getting the second product when the first product is bought
- lift = When the first product is bought, the probability of getting the second product increases by 1.5 (example) times.
It can find hidden relationships, although with less frequency.
- leverage = It is similar to lift. The lift is used because it tends to prioritize values with higher support.
"""

rules.sort_values("lift", ascending = False).head()

# Filtering is done by taking the intersection of some metrics.

############################################
# Making Product Suggestions to Users at the Basket Stage
############################################

product_id = 22492
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)
# Sort products by lift

recommendation_list = []

"""
- Browse through the first product sets, if the product_id is found, go to the other product in the index where I found it and show that product.
- The lift of the product I want to buy is expected to be the highest
"""
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
# convert antecedents(binary values) to list-> so it becomes int

recommendation_list[0:2]

check_id(df, 21987)
check_id(df, 23235)
check_id(df, 22477)

check_id(df, recommendation_list[0])

# Since it is an Antecident tuple, it is converted to a list and a search is made within the list.
# rec_count = how many observation units should be printed

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

# gives recommendations by product
arl_recommender(rules, 21987, 1)
arl_recommender(rules, 23235, 2)
arl_recommender(rules, 22477, 3)

