FROM public.ecr.aws/lambda/python:3.7

COPY requirements.txt ./
RUN python3.7 -m pip install -r requirements.txt -t .

COPY ./small_car_recommendation_data_aligned.csv ./

COPY ./nearest_neighbors_model2.pkl ./

COPY ./lambda_function.py ./
RUN python lambda_function.py



CMD ["lambda_function.recommend"]