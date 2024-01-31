import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import json
import os

def load_features(credit_score, id_type, marital_status, gender, employment_type):
    print("INSIFE LOAD FUNC" )
    df = pd.read_csv('/var/task/small_car_recommendation_data_aligned.csv')
    print("DATA loaded success")

    features = ['CreditScore', 'IDType', 'BPKBOwnerMaritalStatus', 'Gender', 'EmploymentType', 'CarMake']
    df_features = df[features]
    
    # # Fill missing values without inplace modification
    df_features = df_features.fillna(0)
    
    # # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df_features, columns=['CreditScore', 'IDType', 'BPKBOwnerMaritalStatus', 'Gender', 'EmploymentType', 'CarMake'])
    

    with open('/var/task/nearest_neighbors_model2.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    print("model loaded ", loaded_model)
    # Get the column names after one-hot encoding
    encoded_columns = df_encoded.columns

    # print("Encoded Columns: ", encoded_columns)

    
    input_data = pd.get_dummies(pd.DataFrame({'CreditScore': credit_score, 'IDType': [id_type], 'BPKBOwnerMaritalStatus': [marital_status], 'Gender': [gender], 'EmploymentType': [employment_type]}), columns=['IDType', 'BPKBOwnerMaritalStatus', 'Gender', 'EmploymentType', 'CreditScore'])

    missing_cols = set(encoded_columns) - set(input_data.columns)
    for col in missing_cols:
        # print("Missing COL: ", col)
        input_data[col] = 0

    # Reorder columns to match the order during training
    input_data = input_data[encoded_columns]

    # Find closest neighbors
    distances, indices = loaded_model.kneighbors(input_data)

    # Get recommendations
    recommendations = []

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for credit score {}:'.format(credit_score))
        else:
            recommended_car_make = df.iloc[indices.flatten()[i]]['CarMake']
            recommended_car_type = df.iloc[indices.flatten()[i]]['CarType']
            recommendations.append((recommended_car_make, recommended_car_type))
            print(f"{i}. Car Make: {recommended_car_make}, Car Type: {recommended_car_type} \n")

    return recommendations

    # Create NearestNeighbors model
    # model = NearestNeighbors(n_neighbors=5)
    # model.fit(df_encoded[encoded_columns])

    # with open('nearest_neighbors_model2.pkl', 'wb') as model_file:
    #     pickle.dump(model, model_file)
 

def recommend(event, context):
    credit_score = 468
    id_type = 'Passport'  
    marital_status = 'Passport'  
    gender = 'Other'  
    employment_type = 'Full Time'  
    
    print(load_features(credit_score, id_type, marital_status, gender, employment_type))

