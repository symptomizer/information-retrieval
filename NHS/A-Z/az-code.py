import requests
import json
import time
import string
from datetime import date

with open('../nhskey.txt', 'r') as file:
   key = file.read()

with open('lastChecked.txt', 'r') as file:
   lastChecked = file.read()

main_url = f'https://api.nhs.uk/conditions/'
main_payload = {'subscription-key':key, 'synonyms':'true'}

call_counter = 1

def initialise(call_counter, key):
    
    # list of all entries 
    all_entries = {}

    for letter in ['v', 'w', 'x', 'y', 'z']:

        checkCalls(call_counter)
       
        print(f'Current Category: {letter}')

        main_payload['category'] = letter
        main_response = requests.get(main_url, params=main_payload)
        call_counter += 1

        current_cat_data = json.loads(main_response.text)
        current_payload = {'subscription-key':key}
        entry_count = 1
        for entry in current_cat_data['significantLink']:

            print(f'Category {letter} Entry Number: {entry_count}')

            checkCalls(call_counter)

            current_url = entry['url']
            current_response = requests.get(current_url, params=current_payload)
            
            call_counter += 1

            all_entries[entry['name']] = current_response.text

            entry_count += 1

        print(f'Category {letter} is complete.')
    
    with open('v-z.json', 'w') as json_file:
        json.dump(all_entries, json_file)

    updateLastChecked()

# Because NHS is dumb
def mergeFiles():

    merged_json = {}

    with open('a.json', 'r') as a_file:
        merged_json.update(json.load(a_file))

    with open('b-d.json', 'r') as bd_file:
        merged_json.update(json.load(bd_file))

    with open('e-h.json', 'r') as eh_file:
        merged_json.update(json.load(eh_file))

    with open('i-u.json', 'r') as il_file:
        merged_json.update(json.load(il_file))

    with open('v-z.json', 'r') as vz_file:
        merged_json.update(json.load(vz_file))
    
    # Output merged file
    with open('nhs_az.json', 'w') as json_file:
        json.dump(merged_json, json_file)

# Check we're not about to hit the max calls threshold
def checkCalls(call_counter):
    if (call_counter % 10 == 0):
        print("Sleep timer starting")
        time.sleep(60)
        print("Sleep timer complete")

# Update last checked date to keep track of when we last updated our dataset
def updateLastChecked():

    today = date.today()
    lastChecked = today.strftime("%Y-%m-%d")
    
    with open('lastChecked.txt', 'w') as file:
        file.write(lastChecked)

# Update our dataset with any new entries
def update(call_counter):
    
    updated_entries = {}

    with open('nhs_az.json', 'r') as json_file:
        all_entries = json.load(json_file)
    
    print(f'Last updated {lastChecked}')

    update_payload = {'subscription-key':key, 'startDate':lastChecked, 'orderBy':'dateModified'}
    update_response = requests.get(main_url, params=update_payload)
    call_counter += 1

    update_data = json.loads(update_response.text)

    for entry in update_data['significantLink']:
        print(f'Updating entry number: {call_counter}')
        checkCalls(call_counter)

        current_url = entry['url']
        current_payload = {'subscription-key':key}
        current_response = requests.get(current_url, params=current_payload)

        call_counter += 1

        all_entries[entry['name']] = current_response.text
        updated_entries[entry['name']] = current_response.text
        print('Updated successfully')

    updateLastChecked()

    with open('nhs_az.json', 'w') as json_file:
        json.dump(all_entries, json_file)
    
    with open('nhs_az_updated_entries.json', 'w') as json_file:
        json.dump(updated_entries, json_file)
    
    print ('All entries updated successfully')

def convertToTxt():
    with open('nhs_az.json', 'r') as f:
        txt = json.load(f)
    with open('nhs_az.txt', 'w') as f:
        f.write(json.dumps(txt))
update(call_counter)