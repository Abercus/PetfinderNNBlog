## PetFinder.my Adoption Prediction
### Introduction

This blog post introduces our team’s solution to the Neural Networks (LTAT.02.001) project, which is based on the Kaggle competition [“PetFinder.my Adoption Prediction”](https://www.kaggle.com/c/petfinder-adoption-prediction). The project’s goal is to predict how fast the pet is adopted (if adopted at all), using the data from the Malaysian animal welfare platform PetFinder.my. The dataset includes the general data (the pet’s size, color, breed, name, etc), the images, image metadata (supplementary information for the images from Google’s Vision API analysis), and the sentiment analysis (from Google’s Natural Language API analysis for pet descriptions). The required output for the model is a categorical label from 0-4. The submissions for the contest are graded by making use of quadratic weighted kappa, which measures agreement between two raters - 0 if there is no agreement at all and 1 if there is complete agreement between raters.

### Data Overview and Exploration
The first step is to try and understand the data we are working with. For this we consult the documentation provided at the competition’s website. We provide descriptions for each of the features and try to describe them by making use of visualizations. By doing this we hope to get some ideas on more informative features within the dataset.

PetFinder.my provides 14,993 samples for training, where each sample has 24 features + n images (depict a dog or a cat) + metadata + sentiment info. There are a total of 58,311 images per training set, where some images (see Image 1) illustrate multiple animals.

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Image 1. Example of the case, where one image file has multiple images of the animal.

Provided by the competition holder are following files: 
- _train.csv_ - consists of training data, this is the set of data with target labels, which we can use to train our models on;
- _test.csv_ - consists of test data, without target label, this will be used for Kaggle submission;
- Image metadata files - set of json files, with output of image analysis provided by the contest holder; 
- Sentiment data files - set of json files, with output of sentiment analysis on _Description_ field.


Additionally there are files to map integer values within _train.csv_ which represent categories to strings: _breed_labels.csv_ (BreedId to a name of the breed), _color_labels.csv_ (color id to a color), _state_labels.csv_ (state id to a state). 



In _train.csv_ the following features are included: 
- PetID - ID representing a pet, this can be used to map sentiment analysis results and pet images to a specific pet;
- AdoptionSpeed - This is the target label used for training and prediction, it takes an integer value between 0 to 4. **0** means the pet was adopted on the day of it being listed within their system, **1** means that the pet was adopted between within 1-7 days,**2** within 8-30 days, **3** within 31-90 days. **4** means that the pet was not adopted after 100 days of being listed;
- Type - represents type of a pet, where 1 stands for Dog and 2 for Cat;
- Name - represents name of a pet;
- Age - pet’s age in months;
- Breed1 - primary breed;
- Breed2 - secondary breed, this is missing if pet is a purebred;
- Gender - pet’s gender, where 1 is Male, 2 is Female and 3 is Mixed (because listing can include several pets);
- Color1-3 - Colors of pets;
- MaturitySize - pet’s size, where 0 means is unspecified, 1 stands for a small pet, 2 for medium, 3 for large and 4 for extra large;
- FurLength - pet’s fur length, 0 is not specified, 1 is short, 2 is medium and 3 is long;
- Vaccinated - pet’s vaccination status, 1 means vaccinated, 2 unvaccinated and 3 means unsure;
- Dewormed - pet’s deworm status, 1 means dewormed, 2 not dewormed and 3 unsure;
- Sterilized - pet’s sterilization status, where 1 means spayed / neutered, 2 not spayed / neutered  and 3 unsure;
- Health - pet’s health condition, where 1 is healthy, 2 minor injuries,  3 serious injuries and 4 not specified;
- Quantity - count of pets within the listing;
- Fee - pet’s adoption fee, where 0 is free;
- State - state within country of Malaysia;
- RescuerID - rescuer’s ID;
- VideoAmt - count of videos within listing;
- PhotoAmt - count of photos within listing;
- Description - free-form text for describing the pet / listing.


Given by the contest holder is output data from sentiment analysis performed using Google's Natural Language API on field _Description_, and in the file there is a score and a magnitude for every sentence. Score is within a range of -1 to 1, with positive score indicating the positive nature of a sentence and negative indicating the opposite. The magnitude shows how strong the given sentiment is. Also provided is a summarized magnitude and score for the full text. Those are within zipped files _train_sentiment.zip_ and _test_sentiment.zip_.


For images, the contest provides output data from running all the images through Google's Vision API, which provides Face Annotation, Label Annotation, Text Annotation and Image Properties, giving information on for example where pets are located on pictures and what do they seem to look like. Those are within zipped files _train_images.zip_ and _test_images.zip_.


## Checking for missing values
From the features within _train.csv_, we found that there were 1257 missing values for _Name_ and 12 missing values for _Description_.

## Investigating feature distributions
To better know what’s going on in the data, we performed visualized the distributions of features. We plot the distribution of the features individually and then also look at the distributions by _AdoptionSpeed_ (target). On Figure 2 is plotted the target variable _AdoptionSpeed_.  We first note, that the class of pets adopted within the day of listing is considerably smaller (under 500) compared to every other class (over 3000 to 4200). 

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Figure 2. Distribution of target variable AdoptionSpeed.

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Figure 3. Average number of pets adopted per day within timeframe.

Plotted on Figure 3 is the number of pets adopted per day timeframes. By dividing the number of pets with the size of timeframe (number of days), we can find how many pets were adopted on average on each day. We note that most adoptions happen during the first week of the listing. As time goes on, the number of adoptions per day gets lower.


![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Figure 4. Counts of dogs (1) and cats (2) within the training dataset.

Plotted on Figure 4 is the number of dogs and cats, we can see that the counts of dogs and cats are pretty close, with there being some more dogs than cats.

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Figure 5. Distributions of AdoptionSpeeds for dogs and cats separately.

Plotted on Figure 5 are normalized distributions of the target label counts for dogs (1) and cats (2), where we can see that cats are more likely to be adopted and are adopted earlier. 

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Figure 6. Distributions of AdoptionSpeeds and counts of Sterilization status. 1 is spayed / neutered, 2 is not spayed/neutered and 3 means unsure. We can see, that most of the pets are listed as not sterilized, which can be due to costs and additional work needed to be done. From data, oddly enough, it seems that the the non-spayed pets have higher adoption rate compared to sterilized ones.

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Figure 7. 1 - small, 2 - medium, 3 - large, 4 - extra large

We note that most of the pets are medium size and and almost none are extra large. From the distributions, we can note that people prefer to adopt either small or extra large pets, with medium and large being more neglected.

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Figure 8. 1 - short, 2 - medium, 3 - long, 4 - not specified. 

We note that most pets have short fur and least with long fur. From the distributions of AdoptionSpeed we note that people prefer pets with long fur. Medium fur pets are slightly more adopted compared to short fur. There were no pets with unspecified fur length within the data set.  

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Figure 9. 1- Male, 2 - Female, 3 - Mixed

Figure 9 plots the distribution of gender (right) and adoption speed distributions per gender (left). In the training data, there are more female than male dogs. Looking at the distribution of adoption speeds we can see that male pets are slightly more preferred to female pets.

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Figure 10 plots the distribution of health status (right) and adoption speed distributions per health status (left). We note that most of the pets in the data set are healthy (1), with few having minor injuries (2) and only small with major injuries. We can note, that injured pets are less likely to be adopted.

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Figure 11 plots the distribution of pet being dewormed (right) and adoption speed distributions for each dewormed status. Most of the pets are dewormed (1), but there is also plenty which are either not dewormed (2) or not specified (3). We can actually 

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

![Image](https://github.com/Abercus/PetfinderNNBlog/blob/master/00c19f4fa-1.jpg)

Most of the pets are do not have any fee.

Looking at the average fee for each class, we can see that there is not much of a difference:

AdpSpeed | 0 | 1 | 2 | 3 | 4 
-----|-----|-----|-----|-----|----- 
Avg Fee |22.086|21.822|21.582|20.151|21.315 

### Baseline modeling
To evaluate how useful our neural networks are, we compared them to some baseline models, namely the **Random Forest classifier** (RFC) and **Gaussian Naïve Bayes** (GNB). RFC was selected because of its speed and good performance in many machine learning tasks, while GNB is a good probabilistic baseline for...  RFC performed the best on the raw … data and thus, was selected as the baseline method for further tests. 

The sklearn implementations of both methods were used in this project. We tested various values of the _n_estimators_ and _max_depth_ parameters, the first specifying the number of trees in the forest and the second setting the maximum depth of a tree in the forest. According to the test cohen-kappa score, accuracy and cross-validation accuracy we found the best _n_estimators_ to be 60 and _max_depth_ 15. These are kept constant through the rest of the project. 

In addition, we ran the data through the **SelectKBest** method before applying RFC, testing if feature selection would significantly improve the scores on … data so that we could eliminate some redundant features immediately. In the end, testing different _k_ values and scoring functions did not give notable differences in the results and thus only the RFC model was kept from the pipeline.

### Feature engineering
From our initial tests it quickly became clear that only using some basic techniques like normalization and one-hot-encoding would not give us good results. Therefore, we implemented various feature engineering methods to extract more useful information from the given data.

## TF-IDF
TF-IDF is a metric to evaluate the significance of a word in some document which belongs to a group of documents. TF-IDF considers both the frequency of the word in the particular document (TF, term frequency) and the inverse document frequency (IDF) which decreases the importance of words that frequently appear in all documents (e.g very common words, like articles). 

The intention is to use this metric to determine the most important words in the pet descriptions for every adoption speed category. If there even exists a small set of words that represent a specific category, it could provide useful information to help differentiate the samples of different categories.

There are different variants of the formula, the ones used in this project are the following:
….
https://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html https://skymind.ai/wiki/bagofwords-tf-idf

The normalized variant of the TF formula was used with K = 0.5. 
In this case, a document comprised all of the descriptions in the given category (adoption speed) and thus there were altogether 5 documents. The TF-IDF score was calculated for every word in a document to determine the words that best describe each adoption speed. To ease the process, different preprocessing methods were applied to the texts, including lemmatization and name extraction (i.e. eliminating names from texts). 

The 10 best features were selected from every category and used as features. The feature values were calculated as the number of times the word appeared in the given description. 






### Remove:
Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Abercus/PetfinderNNBlog/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.



You can use the [editor on GitHub](https://github.com/Abercus/PetfinderNNBlog/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.


