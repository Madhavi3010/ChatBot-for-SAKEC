from apriori import apriori
import csv

# Read transactions from CSV file
transactions = []
with open('iris.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        transactions.append(row)

# Convert transactions to a list of sets
transaction_sets = [set(transaction) for transaction in transactions]

# Perform Apriori algorithm
min_support = 0.5  # Minimum support threshold
frequent_itemsets = apriori(transaction_sets, min_support)

# Print frequent itemsets
for itemset, support in frequent_itemsets:
    print(itemset, support)
