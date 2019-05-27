## PetFinder.my Adoption Prediction
### Introduction

This blog post introduces our team’s solution to the Neural Networks (LTAT.02.001) project, which is based on the Kaggle competition “PetFinder.my Adoption Prediction” . The project’s goal is to predict how fast the pet is adopted (if adopted at all), using the data from the Malaysian animal welfare platform PetFinder.my. The dataset includes the general data (the pet’s size, color, breed, name, etc), the images, image metadata (supplementary information for the images from Google’s Vision API analysis), and the sentiment analysis (from Google’s Natural Language API analysis for pet descriptions). The required output for the model is a categorical label from 0-4. The submissions for the contest are graded by making use of quadratic weighted kappa, which measures agreement between two raters - 0 if there is no agreement at all and 1 if there is complete agreement between raters.

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


