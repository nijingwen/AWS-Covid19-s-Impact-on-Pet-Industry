# Amazon Final Project
This project is a collaborative work by Jingwen Ni, Haohan Shi, Lu Zhang, and Zhe Zhang. This README will first present a brief [**Introduction**](#introduction) to the project and state its significance. We will then focus on the [**Data**](#data), [**Method: Machine Learning Model**](#method-machine-learning-model), and [**Method: Topic Modeling**](#method-topic-modeling) sections and present our [**Discussion and Limitation**](#discussion-and-limitation). <br><br>
In terms of large-scale data techniques, we have used **lambda function** and **step function** to web scrape Amazon data, we have used **Dask** to conduct EDA, and we have used **Spark** to perform feature engineering (include sentiment analysis), machine learning, and topic modeling.

## Introduction
Covid-19, later renamed Coronavirus, gripped China since the beginning of 2020 (Qiu et al., 2020). By the end of November 2021, there were more than 63 million reported cases and 1.4 million deaths worldwide (Brodeur et al., 2021). The degree of the finicial crash was more severe than the crisis in 2008(Castillo, Melin 2020). A study suggested that from the European Commission's Spring 2020 Monetary Conjecture report, the total national output will contract profoundly: 7.5% for the EU, 4% for Poland, 9% for Italy, France, and Spain (Lai et al., 2021). Covid-19 spread rapidly worldwide and brought an unignorable crisis in all industries, including the pet industry.<br>

Since Covid-19 is a respiratory infectious disease, countries have published several policies about social distance and quarantine to reduce Covid-19 infections. An [online survey](https://www.statista.com/statistics/1191540/quality-time-with-pets-by-generation-due-to-covid-19-us/) on 2,007 respondents in November 2020 shows that approximately 70% of Americans from all generations spend more time with their pets as a result of social distancing regulations. A previous study shows that the Relative Search Volume (RSV) for both god and cat adoption in 2020 increased by up to 250% compared with the RSV in the same period in 2019 (Ho et al., 2021). [U.S.News](https://www.usnews.com/news/healthiest-communities/articles/2020-03-24/people-are-stepping-up-to-foster-pets-during-the-coronavirus-pandemic) also reported an increase in pet adoption due to the pandemic. Two of our group members foster pets for the first time in our lives during Covid (Bento on the right and Fortune on the left) and get huge emotional support from our pets. <br>
<img src="https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/bento.png" width=30% height=30%> <img src="https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/fortune.png" width=22.5% height=22.5%> <br>

We thus want to explore COVID-19's impact on the pet industry. <br>

We think this envisioned study is critical to understanding how consumer behavior is influenced by major events such as the Covid-19. This study can be further applied to other industries. By applying large-scale computing, we minimized the computational time needed for each industry. Therefore other researchers can use our code and analyze the whole market in a reasonable amount of time. <br>

## Data
### Amazon Data
lambda function ([amazon_scraper.zip](https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Code/amazon_scraper.zip)) and step function ([amazon_scraper_sfn_setup.py](https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Code/amazon_scraper_sfn_setup.py))<br>
The resulting data is too be to be uploaded on GitHub. Boto3 notebook is in [scraper](https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Code/amazon_scraper.ipynb). [This](https://drive.google.com/file/d/1tjZvfS2l617xd47DBTyPRgryOwpBYPeu/view?usp=sharing) is the Google Drive Link to the data. 

We used Beautiful Soup to write scraper for Amazon products within the pet section. The scraper can automatically go through all products on each page and store each product's information as a JSON file in AWS S3. To speed the scraping process up, we parallel the process using AWS lambda and Step Functions. <br>

For each product, we scraped information about: <br>
- product id (String)
- product title (String)
- overall star rating (Double)
- number of rating (Integer)
- price (Double)
- review text (String)
- time of the review (Date)
- location of the review (String)
- star rating of this review (Integer)
- number of helpful votes (Integer) <br>

Each instance/observation in the data is a unique review of pet-related amazon products. We have in total more than 330,000 reviews for 1161 distinct products.

### Covid Data
We got the aggregated Covid case data from [the New York Times](https://github.com/nytimes/covid-19-data/blob/master/us.csv). Since Amazon only published review locations on the country level, we can only use data on this level although. Features of this data include: <br>
- date
- aggregated number of Covid cases
- aggregated number of Covid death cases <br>

Each instance/observation in the data is a unique day since 2020-01-21.

### Data Cleaning
[cleaning.ipynb](https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Code/cleaning.ipynb) <br>

In this file, we first cleaned the `Amazon Data` and cast each feature to its corresponding type. <br>
We then created the increased number of Covid cases per day and the increased number of Covid death per day for the `Covid Data`. We also cast each feature to its corresponding type. <br>

We then merged `Covid Data` with `Amazon Data`. Reviews wrote before Covid have `Covid Data` related features (`cases`, `death`, `increased cases`, and `increased death`) as 0. 

### Data Visulization
[chartplot.ipynb](https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Code/chartplot.ipynb) <br>

We visualized the variables in the datasets using Dask. Here are some visulizations that we think worth noticing (Other visulizations are performed yet not presented here. All visualizations are stored [here](https://github.com/lsc4ss-s22/final-project-amazon/tree/main/Graph/visiual_chart)): <br>

<img src="https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/visiual_chart/Comments%20Count%20by%20Year.png" width=50% height=50%> <br>
The graph describes the changes in the number of comments on the website, which increased gradually from 2004 until it reached its peak in 2021. 5 months have passed in 2022, and the number of comments almost reaches the level of 2020. This can either caused by the increase in the amount of new product, or the increase in the number of customers who are more willing to leave comments under the product. Either explanation indicates the prosperity in the pet industry. 

<img src="https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/visiual_chart/Avg%20price%20per%20year.png" width=50% height=50%> <br>
The chart shows the change of average price of products in different years. As can be seen from the chart, although the average price of products has been declining since 2014, the price increased slightly until the pandemic. Thus, the epidemic has increased people's demand for pet products.

<img src="https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/visiual_chart/Relation%20between%20price%20and%20star.png" width=50% height=50%> <br>
According to the modeling of price and review star, we found that the higher the price, the worse the product's evaluation. But it also has to do with the lack of samples for higher-priced products.

<img src="https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/visiual_chart/Distribution%20of%20products'%20price.png" width=50% height=50%> <br>
Most pet related product are not expansive and the distribution is as expected. However, this distribution is right skewed, suggesting that we need to normalize the data before using it. 

## Method: Machine Learning Model
[machine_learning.ipynb](https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Code/machine_learning.ipynb) <br>

To test our hypothesis that the reviews of pets-related products is significantly different after major events such as the Covid-19, we decided to train machine learning models. <br>

### Target Variable
Originally, we wanted to set the target variable as the diversity of the reviews per day. We first classify text with [pre-trained sentiment analysis model on IMDB reviews](https://nlp.johnsnowlabs.com/2021/01/15/analyze_sentimentdl_glove_imdb_en.html) and [bert-based emotion recognition](https://nlp.johnsnowlabs.com/2022/01/14/bert_sequence_classifier_emotion_en.html). We test several Spark NLP classification models with some sampled reviews, and these two models yield the most accurate results. Each text is assigned a sentiment (`positive`, `neutral`, or `negative`) and an emotion (`sadness`, `love`, `joy`, `anger`, `fear`, or `surprise`). We assembled the two classification results into one vector. We then calculated the mean of pair-wise similarity scores between the assembled vector per day. This resulting score represents the diversity of reviews for that particular day. <br>

However, we faced the following problems: <br>
1) The sentiment analysis requires large memory. We opened new personal and selected instances with large memory to perform this sentiment analysis, but it's impossible for us either to write the computed data out to S3 or to perform further steps on that set of instances. We eventually have to use m5. When we try to increase the number of m5 cores, the machine learning model can be computed fast, but the sentiment analysis takes forever. We eventually have to perform the most basic sentiment analysis on 2 cores, and train the machine learning models on 10 cores. 
1) Although we are able to covert categorical sentiment classes to numerical features in the Pipeline without explicitly writing it out using `withColumn`, it takes forever for PySpark to compute this step. We thus need to take it out and create separate steps.
2) When iterating through the data and calculating the pair-wise similarity scores, the `collect()` function on large dataset results in the `OutOfMemory` error. This problem is also noted by many others [here](https://sparkbyexamples.com/pyspark/pyspark-collect/). We already used a personal account and used instances with large memory, but the problem remained. <br>

Due to all the constraints we faced, we eventually had to give up our original idea. The implementations of the `pre-trained sentiment analysis model` and the `bert-based emotion recognition` are at the end of the `machine_learning.ipynb`. When we have more funding, we can easily implement those models. <br>

We thus decided to look at if Covid has any influence on product reviews' sentiment. The sentiment is estimated using [a model developed by Vivek Narayanan](https://nlp.johnsnowlabs.com/2021/11/22/sentiment_vivekn_en.html), which requires smaller memory but also produces lower accuracy. 

### Feature Selection
We mainly focus on the COVID-related variables, so we acquired COVID-related deaths and cases from https://github.com/nytimes/covid-19-data/blob/master/us.csv. We have also constructed 2 new features: ``increased deaths`` and ``increased cases``. <br>

We have also included 2 variables that are related to the reivews and products: ``review length`` and ``product price``, and these 2 variables can serve
has control variables in our logistic regression model. <br>

We did not use tf-idf becuase then the weight of other features would be small, and tf-idf can have a high correlation with the emotion lables, which can 
be divergent to our goal of this project. <br>

Here is an example of our dataframe: <br>
<img src=https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/visiual_chart/dataframe.png width=50% height=50%> <br>

We have also normalized the feature matrix, so that the different scales of the features would not have much effect in feature importance.

### Balance Data
Before balancing the data, the counts of each label are shown below:

|sentiment_code|count|
|---|---|
|1|11753|
|0|14654| <br>

After balancing the data, the counts of each label are shown below:

|sentiment_code|count|
|---|---|
|1|11753|
|0|12249|

### Model Selection
In this task, we mainly want to see the importance of the features, so our main model is the **logistic regression model**. <br>
We are also interested in how accurate this prediction can be, so we employed 4 other models to test this feature-label combination, namely **random forest**, **gaussian naive bayes**, **gradient boost tree**, and **linear SVM**. 

### Model Evaluation
Here we only closely evaluate the logistic regression model.
#### Logistic Regression
The AUC of the model is shown below:<br>
<img src=https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/visiual_chart/auc.png width=30% height=30%> <br>

The ROC plot is shown:below:<br>
<img src=https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/visiual_chart/ROC%20plot.png width=50% height=50%>

We can see that the peroformance of the model is really bad. In addition, the coefficient of the features are shown below:


|price|length|cases|deaths|increased cases|increased deaths|
|-|-|-|-|-|-|
|1.0592| -0.1002| 0.0259| -2.287| 1.0431| -31.2387| <br>

We can see that ``increased deaths`` has a pretty strong negative effect on the label. Given that we encoded positive sentiment as 1 and negative sentiment as 0, ``increased deaths`` has a negative effect on the sentiment. This means that when there are more increased deaths, the sentiment of the reviews tends to be negative.. <br>

### Model Comparison
<img src = https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/visiual_chart/comparison.png width=50% height=50%>

We can see that all of the models perform equally badly on the test set. **Random forest** has comparatively high accuracy on the training set. This is
expected because **random forest** is a very complex model, making it very easy to overfit on the training set. <br>

Based on this comparison, we conclude that the sentiment of the reviews cannot be predicted by our features, and **increased deaths** has a negative effect on the reviews' sentiment.



## Method: Topic Modeling
[topic_modeling.ipynb](https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Code/topic_modeling.ipynb) <br>

We then performed LDA Topic Modeling using PySpark to see if topics changed in the pet industry before and after covid. We pre-processed the data using `documentAssembler`, `tokenizer`, `normalizer`, `lemmatizer`, `stopwords cleaner`, and `finisher`. <br>

To determine which words are important, we used TF-IDF vectorizer implemented by first using the `CountVectorizer` and then the `IDF` estimator. <br>
Due to the computational time limit, we set the number of topics to 6 and the maximum iteration to 50. <br>

After the topic models before and after Covid are trained, we used the UDF that converts word ids (the actual output for a topic by a topic model) into the words that describe the derived topics. We looked at the top 10 words for each model and their corresponding termWeights. <br>

The results are visualized through Dask. <br>
![before_wordcloud](https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/visiual_chart/before_wordcloud.png) <br>
![after_wordcloud](https://github.com/lsc4ss-s22/final-project-amazon/blob/main/Graph/visiual_chart/after_wordcloud.png) <br>

Most topics are similar before and after Covid. However, the 5th topic before Covid contains top 10 words: `seat`, `leash`, `diaper`, `car`, `harness`, `cover`, `u`, `velcro`, `strap`, and `belt`. This topic and these words are not in the topics after Covid. These are all out-door related words, meaning that reviews for outdoor activities and related products are largely influenced by Covid. 

## Discussion and Limitation
We collected Amazon data and Covid data. We visualized the data and then performed machine learning and topic modeling to investigate Covid's influence on the pet industry. Our visualizations suggested that the pet industry has been flourishing these years and is likely to continue this increasing trend. The machine learning results suggested that Covid is not significantly influencing people's sentiment towards pet-related products. However, LDA topic modeling suggests that out-door related pet products lose their popularity in the public pet-related discursive field. <br>

As presented in the machine learning model section, we faced constraints with AWS. We implemented a better Pipeline and alternative ideas on a small scale through a personal account. We believe that with more resources in the future, we can improve the accuracy of our model and expand this project easily. This project can also be directly used to understand consumer behavior in other industries. <br>

## Reference
Brodeur, A., Clark, A. E., Fleche, S., & Powdthavee, N. (2021). COVID-19, lockdowns and well-being: Evidence from Google Trends. Journal of Public Economics, 193, 104346. https://doi.org/10.1016/j.jpubeco.2020.104346 <br>
Castillo, O., & Melin, P. (2020). Forecasting of COVID-19 time series for countries in the world based on a hybrid approach combining the fractal dimension and fuzzy logic. Chaos, Solitons & Fractals, 140, 110242. https://doi.org/10.1016/j.chaos.2020.110242 <br>
Ho, J., Hussain, S., & Sparagano, O. (2021). Did the COVID-19 Pandemic Spark a Public Interest in Pet Adoption? Frontiers in Veterinary Science, 8. https://www.frontiersin.org/article/10.3389/fvets.2021.647308 <br>
Lai, H., Khan, Y. A., Thaljaoui, A., Chammam, W., & Abbas, S. Z. (2021). COVID-19 pandemic and unemployment rate: A hybrid unemployment rate prediction approach for developed and developing countries of Asia. Soft Computing. https://doi.org/10.1007/s00500-021-05871-6 <br>
Qiu, Y., Chen, X., & Shi, W. (2020). Impacts of social and economic factors on the transmission of coronavirus disease 2019 (COVID-19) in China. Journal of Population Economics, 33(4), 1127–1172. https://doi.org/10.1007/s00148-020-00778-2 <br>

## Division of Work
Haohan Shi: Data Collection, Machine Learning Models <br>
Lu Zhang: LDA Topic Modelings, Data Cleaning, README <br>
Jingwen Ni: Data Collection and Cleaning, Presentation <br>
Zhe Zhang: Dask Visulizations, Presentation <br>

We also thank Bento and Fortune for their inspiration and emotional support. <br>
