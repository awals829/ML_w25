# Assignment 1: Data discovery and visualization
Due January 31, 2025 at 5 pm. Reasonable requests for extensions will be granted when requested at least 48 hours before the due date.

You may work in groups of 2 or 3. Click [here](https://classroom.github.com/a/QF1v7JX1) to create your group on GitHub Classroom.

## Table of Contents <!-- omit in toc -->
- [Overview](#overview)
- [Dataset details](#dataset-details)
    - [YYC Housing Data](#yyc-housing-data)
    - [Glenmore Traffic Data](#glenmore-traffic-data)
- [Deliverables](#deliverables)
    - [Exploration and model training notebook](#exploration-and-model-training-notebook)
    - ["Production" code](#production-code)
    - [Written response](#written-response)
    - [Code reviews](#code-reviews)
- [Tips](#tips)
- [Marking Scheme](#marking-scheme)

## Overview
Real world data is messy and incomplete in unexpected ways. Often, the information you need is in some kind of text field, or in a totally separate database that needs to be merged in.  While I have done some minimal filtering of the two dataset options, the focus of this assignment is on exploring and preparing the data for use with a machine learning model. 

Choose **one** of the following datasets for your assignment:

1. YYC Housing Data: 2024 assessment values for residential properties in Calgary.

2. Glenmore traffic data: automatic traffic count volumes for 2023 on Glenmore Trail SW at Richard Road, measured in 15-minute increments.

For both datasets, you will be training a regressor to predict a numeric value. You may choose any of the regression models from [scikit-learn's supervised learning modules](https://scikit-learn.org/stable/supervised_learning.html), provided inference (prediction) is fairly fast (i.e. don't use nearest neighbours). If you want to use something other than scikit-learn, just let me know - it's probably fine! Again, the main focus of this assignment is the dataset exploration and processing part.

I have set aside a subset of the data to test your models. The model performance will be evaluated and compared using the [Mean Absolute  Error](https://en.wikipedia.org/wiki/Mean_absolute_error). Note that since these are not practice or training datasets, the results may not be very good! As long as you're in a reasonable range (less than about 100% relative error), your grade will not be affected by the prediction performance.

## Dataset details
Both datasets were downloaded from https://data.calgary.ca/ on January 12, 2025.

### YYC Housing Data

[Source](https://data.calgary.ca/Government/Current-Year-Property-Assessments-Parcel-/4bsw-nn7w/about_data)

I have removed several redundant columns and excluded non-residential properties such as parking spots.

Unlike the California housing dataset example, this contains property details and assessed values for each individual house in Calgary. Your goal will be to try to predict the `ASSESSED_VALUE` based on the other columns of the dataset. Be careful though - there are some weird city-specific codes in there. For example, the `SUB_PROPERTY_USE` column contains the following codes:

| Code   | Description                  |
| ------ | ---------------------------- |
| RE0100 | Residential Acreage          |
| RE0110 | Detached                     |
| RE0111 | Detached with Backyard Suite |
| RE0120 | Duplex                       |
| RE0121 | Duplex Building              |
| RE0201 | Low Rise Apartment Condo     |
| RE0210 | Low Rise Rental Condo        |
| RE0301 | High Rise Apartment Condo    |
| RE0310 | High Rise Rental Condo       |
| RE0401 | Townhouse                    |
| RE0410 | Townhouse Complex            |
| RE0601 | Collective Residence         |
| RE0800 | Manufactured Home            |

Similarly, the `LAND_USE_DESIGNATION` refers to the city's [land use zones](https://www.calgary.ca/planning/land-use/districts.html), which restrict the type of building that can be constructed on a given property. These zones are about to become much simpler, but for now the column exists in the dataset. It's up to you to decide how (or if) to use it.

Feel free to get creative! Are odd-numbered houses worth more than even? Does distance from city centre as the crow flies (lat and lon) really matter? Apply your **domain knowledge** to select and transform your features.

### Glenmore Traffic Data

[Source](https://data.calgary.ca/Transportation-Transit/Traffic-Counts-at-Permanent-stations/vuyp-sbjp/about_data)

Technically, this would be best treated as a time series, where the predicted traffic volume depends on the immediate previous values. However, for the purposes of this assignment, I want you to consider each sample as an independent snapshot.

At a surface level, this dataset is very simple, with only the timestamp, direction (eastbound/westbound), and traffic volume. The challenge is in figuring out how to extract relevant information from that timestamp.

Also challenging is deciding how to sample time series data. Since the measurements are taken at 15 minute intervals, a naive random sample for your test/validation data will result in a lot of samples that are very close to training data, and you will likely end up overfitting. I recommend setting aside your test sample by selecting a chunk (or multiple chunks) of time rather than randomly sampling.

## Deliverables
Your assignment should consist of both a .ipynb notebook (committed with cells rendered) for your exploratory analysis and model training, as well as your "production" code providing a `predict` function.

### Exploration and model training notebook
This notebook should follow the general process outlined in class to do the first four steps of the [ML Project Checklist](https://github.com/ageron/handson-ml3/blob/main/ml-project-checklist.md), plus training of a simple model. The emphasis is on the data exploration and preparation rather than the model itself. Model shortlisting and fine-tuning is not required (though it is allowed if you'd like).

Specifically, this notebook should include:
- loading the data
- setting aside a test set (as appropriate for the problem)
- exploratory visualizations, with comments about your observations
- your preprocessing pipeline
- your model training, either with cross-validation or a set-aside validation dataset
- saving your pipeline + model for production

Some guidelines are provided in the template notebook - feel free to modify as desired.

If you try something and then ultimately don't use it, it's fine to leave it in the notebook. I'd like to see things you thought of and then discarded.

### "Production" code
After deciding on a preprocessing pipeline, training a model, and saving it all to disk, implement the function `predict` in `prod.py`. This function should:
- load the required libraries
- load your model from disk
- apply your preprocessing
- return the predicted values

If you are using Scikit-learn's [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) class to combine preprocessing with your regression model, then this function could be as simple as loading your pipeline and returning `pipeline.predict(data)`.

### Written response
Answer the questions in `reflection.md`. Point form and short responses are fine! If you really hate Markdown, you can add a PDF instead.

### Code reviews
As you are working on this assignment, try to work in either a separate [branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches) or [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks), then submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) to merge the changes into the `main` branch. You and your teammate(s) should [review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/commenting-on-a-pull-request) each others' pull requests with questions and suggestions, then merge in the changes. Any comments that need addressing (such as "this graph needs axis labels") should be addressed with a new commit and merged in. Your comments can either be in the main PR discussion, or on specific lines of the changed files.

If you forgot to work in a separate branch and pushed directly to `main`, you can still add comments to a commit by clicking on the "Commit history" and adding line-specific comments (it's just a bit clunkier).

Since I have never tried asking students to do this before, I have no idea how well it will work. Don't go too crazy with the discussion, I'm mainly just looking for evidence of collaborative work.

## Tips
1. I recommend creating a virtual environment and installing the packages in `requirements.txt`. This will ensure that your code runs on my system:
   ```bash
   python -m venv venv
   pip install -r requirements.txt
   ```
   on a Mac, use `python3` and `pip3` instead.

   If you use any other packages and want to add them to the requirements list, you can update it with:
   ```bash
   pip freeze > requirements.txt
   ```
   (again with `pip3` if you are a Mac user).
2. The [end-to-end ML project](https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb) from the textbook (presented in a condensed form in class) provides examples of *some* data transformation and visualization techniques, but these do not cover all scenarios. You may need to do some additional research to find the right technique for your dataset - in this case, make sure to **cite your sources** with a comment in your code.
3. I have reserved some data for a friendly competition between groups. You might want to test your `predict` function with your own subset of data to make sure the loading and processing behaves in an isolated environment.
4. Make sure to remove the target column from the dataframe before processing! I will be calling `predict` with a dataframe that does not include the target (`VOLUME` or `ASSESSED_VALUE`).

## Marking Scheme
Each of the following components will be marked on a 4-point scale and weighted.

| Component                                       | Weight |
| ----------------------------------------------- | ------ |
| Data exploration (visualizations, observations) | 25%    |
| Preprocessing decisions                         | 25%    |
| Model and training                              | 20%    |
| Written responses                               | 20%    |
| Evidence of collaboration (code reviews)        | 10%    |

| Score | Description                                                            |
| ----- | ---------------------------------------------------------------------- |
| 4     | Excellent - thoughtful and creative without any errors or omissions    |
| 3     | Pretty good, but with minor errors or omissions                        |
| 2     | Mostly complete, but with major errors or omissions, lacking in detail |
| 1     | A minimal effort was made, incomplete or incorrect                     |
| 0     | No effort was made, or the submission is plagiarized                   |
