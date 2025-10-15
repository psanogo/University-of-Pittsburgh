```python
import numpy as np
import pandas as pd
```

**Question 1:** The dataframe called "coinFlips" has two columns; the first represents the results for flipping Coin A and the second column represents the results of flipping Coin B.
In both columns, 1 represents a heads and 0 represents a tails.

Calcuate 
- $\hspace{.25cm}P(A=H)\hspace{.25cm}$ the probability that Coin A is heads, 
- $\hspace{.25cm}P(B=H)\hspace{.25cm}$ the probability the Coin B is heads, 
- and $\hspace{.25cm}P(A=H|B=H)\hspace{.25cm}$ the probability that Coin A is heads given that Coin B is heads (i.e., using the rows in the dataframe).


```python
coinFlips = pd.read_csv("./coinFlipData.csv",index_col=0)
coinFlips.shape
```




    (50, 2)




```python
import pandas as pd

def calculate_coin_probabilities():
    coinFlips = pd.read_csv("coinFlipData.csv", index_col=0)
    
    # Total number of flips
    total_flips = len(coinFlips)
    
    # Probability P(A=H)
    prob_A_heads = coinFlips['Coin A'].mean()
    
    # Probability P(B=H)
    prob_B_heads = coinFlips['Coin B'].mean()
    
    # Probability P(A=H and B=H)
    joint_prob_A_and_B = len(coinFlips[(coinFlips['Coin A'] == 1) & (coinFlips['Coin B'] == 1)]) / total_flips
    
    # Conditional Probability P(A=H | B=H)
    # P(A=H | B=H) = P(A=H and B=H) / P(B=H)
    prob_A_given_B_heads = joint_prob_A_and_B / prob_B_heads
    
    return {
        'P(A=H)': prob_A_heads,
        'P(B=H)': prob_B_heads,
        'P(A=H|B=H)': prob_A_given_B_heads
    }

calculate_coin_probabilities()

```




    {'P(A=H)': 0.6, 'P(B=H)': 0.48, 'P(A=H|B=H)': 0.7083333333333334}




```python
# HIDDEN TEST CELL

```

**Question 2:** What is the p-value associated with the statistical test for the hypothesis that $\hspace{.25cm}P(A=H)>P(B=H)$? And, what is the p-value associated with the statistical test for the hypothesis that $\hspace{.25cm}P(A=H)>P(A=H|B=H)?$


```python
import pandas as pd
import scipy
from statsmodels.stats.proportion import proportions_ztest

def calculate_probabilities():
    coinFlips = pd.read_csv("coinFlipData.csv", index_col=0)

    # Probabilities from Question 1
    prob_A_heads = coinFlips['Coin A'].mean()
    prob_B_heads = coinFlips['Coin B'].mean()
    
    # Calculate counts for proportions_ztest
    successes_A = coinFlips['Coin A'].sum()
    trials_A = len(coinFlips['Coin A'])
    successes_B = coinFlips['Coin B'].sum()
    trials_B = len(coinFlips['Coin B'])
    
    # Test 1: P(A=H) > P(B=H) (one-sided)
    # Perform a z-test for two proportions
    counts = np.array([successes_A, successes_B])
    nobs = np.array([trials_A, trials_B])
    z_stat, p_value_test1 = proportions_ztest(count=counts, nobs=nobs, alternative='larger')

    # Test 2: P(A=H) > P(A=H|B=H) (one-sided)
    # P(A=H|B=H) is not a fixed value but calculated from the data.
    # This comparison effectively tests independence between A and B.
    # We can use a chi-squared test or a Fisher's exact test on the contingency table.
    
    # Build contingency table
    contingency_table = pd.crosstab(coinFlips['Coin A'], coinFlips['Coin B'])
    
    # Perform Fisher's exact test, one-sided ('greater' assumes positive association)
    # The alternative hypothesis P(A=H) > P(A=H|B=H) suggests a negative association, so we use 'less'
    odds_ratio, p_value_test2 = scipy.stats.fisher_exact(contingency_table, alternative='less')
    
    return {
        'p_value_P(A=H)>P(B=H)': p_value_test1,
        'p_value_P(A=H)>P(A=H|B=H)': p_value_test2
    }

calculate_probabilities()

```




    {'p_value_P(A=H)>P(B=H)': 0.11432213117053847,
     'p_value_P(A=H)>P(A=H|B=H)': 0.9641094301401828}




```python
# HIDDEN TEST CELL

```

**Question 3:** The dataframe called "orderData" has two columns. The first column is sales data from Pizza Place 1 and the second columns is sales data from Pizza Place 2. What is the p-value for the hypothesis that the average sale per order is different between the two pizza places? 

Hint: Use a two-sided test.


```python
orderData = pd.read_csv("./orderData.csv",index_col=0)
orderData.shape
```




    (10, 2)




```python
import pandas as pd
import scipy.stats as stats

def compare_sales():
    orderData = pd.read_csv("orderData.csv", index_col=0)
    
    # Extract sales data for each pizza place
    sales_place1 = orderData['Pizza Place 1']
    sales_place2 = orderData['Pizza Place 2']
    
    # Perform a two-sample, two-sided t-test
    # This tests the null hypothesis that the two independent samples have identical average values.
    t_statistic, p_value = stats.ttest_ind(sales_place1, sales_place2, equal_var=False, alternative='two-sided')
    
    return p_value

compare_sales()

```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    File /opt/conda/lib/python3.10/site-packages/pandas/core/indexes/base.py:3803, in Index.get_loc(self, key, method, tolerance)
       3802 try:
    -> 3803     return self._engine.get_loc(casted_key)
       3804 except KeyError as err:


    File /opt/conda/lib/python3.10/site-packages/pandas/_libs/index.pyx:138, in pandas._libs.index.IndexEngine.get_loc()


    File /opt/conda/lib/python3.10/site-packages/pandas/_libs/index.pyx:165, in pandas._libs.index.IndexEngine.get_loc()


    File pandas/_libs/hashtable_class_helper.pxi:5745, in pandas._libs.hashtable.PyObjectHashTable.get_item()


    File pandas/_libs/hashtable_class_helper.pxi:5753, in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'Pizza Place 1'

    
    The above exception was the direct cause of the following exception:


    KeyError                                  Traceback (most recent call last)

    Cell In[8], line 17
         13     t_statistic, p_value = stats.ttest_ind(sales_place1, sales_place2, equal_var=False, alternative='two-sided')
         15     return p_value
    ---> 17 compare_sales()


    Cell In[8], line 8, in compare_sales()
          5 orderData = pd.read_csv("orderData.csv", index_col=0)
          7 # Extract sales data for each pizza place
    ----> 8 sales_place1 = orderData['Pizza Place 1']
          9 sales_place2 = orderData['Pizza Place 2']
         11 # Perform a two-sample, two-sided t-test
         12 # This tests the null hypothesis that the two independent samples have identical average values.


    File /opt/conda/lib/python3.10/site-packages/pandas/core/frame.py:3804, in DataFrame.__getitem__(self, key)
       3802 if self.columns.nlevels > 1:
       3803     return self._getitem_multilevel(key)
    -> 3804 indexer = self.columns.get_loc(key)
       3805 if is_integer(indexer):
       3806     indexer = [indexer]


    File /opt/conda/lib/python3.10/site-packages/pandas/core/indexes/base.py:3805, in Index.get_loc(self, key, method, tolerance)
       3803     return self._engine.get_loc(casted_key)
       3804 except KeyError as err:
    -> 3805     raise KeyError(key) from err
       3806 except TypeError:
       3807     # If we have a listlike key, _check_indexing_error will raise
       3808     #  InvalidIndexError. Otherwise we fall through and re-raise
       3809     #  the TypeError.
       3810     self._check_indexing_error(key)


    KeyError: 'Pizza Place 1'



```python
# HIDDEN TEST CELL

```
