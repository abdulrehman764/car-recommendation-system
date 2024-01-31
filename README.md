# Lambda Function - Machine Learning based Car Recommendation System
 
## Overview
This repository contains a serverless AWS Lambda function for small car recommendations based on a k-nearest neighbors machine learning model. The application utilizes Python 3.7 and leverages the scikit-learn library for machine learning functionalities. The model is trained on a dataset with features such as Credit Score, ID Type, Marital Status, Gender, Employment Type, and Car Make.

## Functionality
- **load_features:** This function loads the required dataset and one-hot encodes categorical columns. It then loads a pre-trained k-nearest neighbors model from a pickled file (`nearest_neighbors_model2.pkl`).
- **recommend:** The main Lambda function (`recommend`) provides an example of how to use the `load_features` function to get recommendations based on input parameters such as credit score, ID type, marital status, gender, and employment type.

## Docker Configuration
The application is packaged as a Docker image using the official AWS Lambda Python 3.7 base image. The required dependencies specified in `requirements.txt` are installed within the image. The Dockerfile also copies the dataset (`small_car_recommendation_data_aligned.csv`) and the pre-trained model (`nearest_neighbors_model2.pkl`) into the image.

## Usage
To deploy this Lambda function on AWS Lambda, build and push the Docker image to a container registry, such as Amazon ECR. The Lambda function is configured to execute the `recommend` function.

## Requirements
- Python 3.7
- Pandas 1.3.5
- Scikit-learn 1.0.2

## How to Run Locally
To run the Lambda function locally, ensure Docker is installed, and then build and run the Docker image.

```bash
docker build -t small-car-recommendation-lambda .
docker run small-car-recommendation-lambda
