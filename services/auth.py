
import requests
from ntt_secrets import NTT_ID, NTT_SECRET

def authenticate():
    '''
    return token after authentication request
    '''

    auth_url = 'https://api.ntth.ai/v1/auth/appLogin'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        # Create a new file ntt_secrets.py and put your own credentials there
        "id": NTT_ID,
        "secret": NTT_SECRET
    }

    try:    
        response = requests.post(auth_url, json=data, headers=headers)
    except Exception as e:
        print(e)
        exit()

    if response.status_code == 200:
        print("auth passed")
    
    token = response.json()['token']


    return token


    messages = [        
            {
            "role": "system",
            "content": "You are an AI assistant that helps answering questions"
            },
            ]

    chat_url = 'https://api.ntth.ai/v1/chat'

    headers = {
    'Accept': 'text/event-stream' ,
    'Content-Type': 'application/json',
    'Authorization': token 
    }
    data =  {
    "id": NTT_ID,
    "modelId": "cc5bab32-9ccf-472b-9d76-91fc2ec5b047",
        "messages": None,
        "maxTokens": 2000,
        "stream": True,
        "presencePenalty": 0,
        "stop": None,
        "frequencyPenalty": 0,
        "topP": 0.95,
        "temperature": 0.7,
        "safeGuardSettings": {
            "status": "monitor",
            "configurationStyle": "easy",
            "configuration": {
            "moderation": "one",
            "personalInformation": "one",
            "promptInjection": "one",
            "unknownLinks": "one"
            }
        },
        "locale": "en"
    }