import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#load data
df=pd.read_csv('data/supermarket_sales.csv')

#display the first few rows
df.head()

#check for missing values
df.isnull().sum()

#drop rows with missing values if any
df.dropna(inplace=True)

#summary Statistics
df.describe()

#sales by product line
plt.figure(figsize=(10,6))
sns.barplot(x='Product Line',y='Total',data=df,estimator=sum)
plt.title('Total Sales by Product Line')
plt.xticks(rotation=45)
plt.show()

#Analyze sales by payment method
plt.figure(figsize=(10,6))
sns.countplot(x='Payment',data=df)
plt.title('Sales Count by Payment Method')
plt.show()

#Identify the city with highest sales
city_sales=df.groupby('City')['Total'].sum().reset_index()
plt.figure(figsize=(8,5))
sns.barplot(x='City',y='Total',data=city_sales)
plt.title('Total Sales by City')
plt.show()

#identify which productline has the highest sales
#group data by 'Product Line' and calculate the total sales for each
product_sales=df.groupby('Product Line')['Total'].sum().reset_index()

#sort by total sales in descending order
product_sales=product_sales.sort_values(by='Total',ascending=False)

#Display the Product Lines with highest sales
print(product_sales)

#Visualize the product Sales
plt.figure(figsize=(10,6))
sns.barplot(x='Total',y='Product Line',data=product_sales,palette='viridis',hue='Product Line', legend=False)
plt.title('Total sales by product line')
plt.xlabel('Total Sales')
plt.ylabel('Product Line')
plt.show()

#Analyzing Sales by Gender
#Group data by 'Gender' and calculate the total sales
gender_sales=df.groupby('Gender')['Total'].sum().reset_index()

#display sales by gender
print(gender_sales)

#visualize sales by gender
plt.figure(figsize=(8,5))
sns.barplot(x='Gender',y='Total',data=gender_sales,palette='viridis',hue='Gender', legend=False)
plt.title('Total Sales by Gender')
plt.xlabel('Gender')
plt.ylabel('Total Sales')
plt.show()

#Analyzing Sales by customer type
#group data by 'customer type' and calcutta the total sales
customer_type_sales=df.groupby('Customer type')['Total'].sum().reset_index()

#Display sales by customer type 
print(customer_type_sales)

#Visualize sales by customer type
plt.figure(figsize=(8,5))
sns.barplot(x='Customer type',y='Total',data=customer_type_sales,palette='viridis',hue='Customer type', legend=False)
plt.title('Total sales by Customer Type')
plt.xlabel('Customer Type')
plt.ylabel('Total Sales')
plt.show()

#Look at average sales per transaction and how it varies across cities
#Group data by 'City' and calculate the average sales per transaction
average_sales_per_city=df.groupby('City')['Total'].mean().reset_index()

#Display average sales per transaction by city
print(average_sales_per_city)

#Visualize average sales per transaction across cities
plt.figure(figsize=(10,6))
sns.barplot(x='Total',y='City',data=average_sales_per_city,palette='viridis',hue='City', legend=False)
plt.title('Average Sales per transaction by city')
plt.xlabel('Average Sales')
plt.ylabel('City')
plt.show()

#Building Predictive Model
#Feature Engineering
df['Month']=pd.to_datetime(df['Date']).dt.month

#Select feature and target
x=df[['Quantity','Unit price','Month']]
y=df['Total']

#split the data
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Train a simple Linear Regression Model
model=LinearRegression()
model.fit(X_train,y_train)

#Model evaluation
score=model.score(X_test,y_test)
print(f"Model R^2 Score:{score}")

