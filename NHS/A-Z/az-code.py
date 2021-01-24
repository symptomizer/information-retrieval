import requests
import pandas as pd
import json
import time
import string
from pprint import pprint

with open('../nhskey.txt', 'r') as file:
   key = file.read()

url = f'https://api.nhs.uk/conditions/'
payload = {'subscription-key':key, 'category':'a', 'synonyms':'true'}

def main():

    call_counter = 1
    
    response = requests.get(url, params=payload)
    call_counter += 1
    json_data = response.json()

    main_data = pd.json_normalize(json_data['significantLink'])

    for letter in string.ascii_lowercase:

        if letter == 'a':
            continue

        print(call_counter)

        if (call_counter % 10 == 0):
            time.sleep(60)

        payload['category'] = letter

        response = requests.get(url, params=payload)
        call_counter += 1

        with open("test.txt", "w") as text_file:
            text_file.write(response.text)
    
        json_data = response.json()
        current_data = pd.json_normalize(json_data['significantLink'])

        main_data = main_data.append(current_data, ignore_index = True)

    main_data.to_csv('nhs_az.csv')

main()
