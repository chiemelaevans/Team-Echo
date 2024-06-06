# Iris Flower Classification using Logistic Regression, Decision Tree, Random Forest, and Naive Bayes

## Overview
In this machine learning project, we utilized well-known classification methods—Logistic Regression, Decision Tree, Random Forest, and Naive Bayes—to categorize iris flowers based on their unique characteristics. Using the Iris dataset, which includes measurements like sepal length, sepal width, petal length, and petal width, we aimed to classify iris flowers into three distinct classes: Iris-Setosa, Iris-Versicolor, and Iris-Virginica.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Description
This project involves the classification of iris flowers using four different machine learning algorithms. We start by importing necessary libraries and reading the dataset. Then, we perform exploratory data analysis (EDA), preprocess the data, and apply various classification algorithms. Finally, we evaluate the performance of each model and compare their accuracy.

## Dataset
The Iris dataset, which is publicly available on the UCI Machine Learning Repository, is used in this project. It includes 150 samples of iris flowers, with four features: sepal length, sepal width, petal length, and petal width. The target variable is the species of the iris flower, which can be one of three classes:
- Iris-Setosa
- Iris-Versicolor
- Iris-Virginica

## Installation
To run this project, you need to have Python and Jupyter Notebook installed. Additionally, you need the following Python libraries:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install these dependencies using pip:
```bash

pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

### Clone the repository
```bash
git clone https://github.com/Nnamdi92/iris-classification.git
```

### Navigate to project directory
```bash
cd iris classification
```

Open jupyter notebook
```bash
jupyter notebook "Iris Project.ipynb"
```

Run the notebook cells sequentially

## Models Used
The following models were used:

**Logistic Regression**: A simple yet effective linear model for binary and multiclass classification.

**Decision Tree**: A non-linear model that splits the data into branches to make decisions.

**Random Forest**: An ensemble method that uses multiple decision trees to improve classification accuracy.

**Naive Bayes**: A probabilistic model based on Bayes' theorem, assuming independence between features.

## Results
The models were evaluated using metrics such as accuracy, precision, recall, and F1-score. The performance comparison shows that each model has its strengths and weaknesses depending on the feature relationships and class separability in the dataset.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. You can also open issues for any bugs or feature requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.






