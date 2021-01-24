import requests
import pandas as pd
import json
import time

with open('azkey.txt', 'r') as file:
   key = file.read()

url = f'https://api.nhs.uk/conditions/'
payload = {'subscription-key':key, 'page':1}

def main():

    call_counter = 1
    
    response = requests.get(url, params=payload)
    call_counter += 1
    json_data = response.json()
    main_data = pd.json_normalize(json_data['significantLink'])

    while response.status_code == 200:

        if (call_counter % 10 == 0):
            time.sleep(60)

        payload['page'] = payload.get('page') + 1
        print(payload['page'])

        response = requests.get(url, params=payload)
        call_counter += 1
        json_data = response.json()
        current_data = pd.json_normalize(json_data['significantLink'])

        main_data = main_data.append(current_data)

    main_data.to_csv('nhs_az.csv')

main()
