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






You can use the [editor on GitHub](https://github.com/Abercus/PetfinderNNBlog/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.




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
