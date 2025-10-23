# test_llm_detailed.py
import json
from services.auth import authenticate
from ntt_secrets import NTT_ID

def test_api_directly():
    """Test API with minimal request"""
    import requests
    
    print("Testing NTT API directly...")
    print(f"ID: {NTT_ID}")
    
    # Get token
    try:
        token = authenticate()
        print(f"✓ Token obtained: {token[:20]}...")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return
    
    # Test 1: Minimal request
    print("\n" + "="*80)
    print("TEST 1: Minimal Request")
    print("="*80)
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': token
    }
    
    # Try different payload variations
    payloads = [
        # Payload 1: Full original
        {
            "id": NTT_ID,
            "modelId": "6c26a584-a988-4fed-92ea-f6501429fab9",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"}
            ],
            "maxTokens": 100,
            "stream": False,
            "presencePenalty": 0,
            "stop": None,
            "frequencyPenalty": 0,
            "topP": 0.95,
            "temperature": 0.2,
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
        },
        # Payload 2: Minimal (without optional fields)
        {
            "id": NTT_ID,
            "modelId": "6c26a584-a988-4fed-92ea-f6501429fab9",
            "messages": [
                {"role": "user", "content": "Say hello"}
            ],
            "maxTokens": 100,
            "stream": False
        },
        # Payload 3: Without stream
        {
            "id": NTT_ID,
            "modelId": "6c26a584-a988-4fed-92ea-f6501429fab9",
            "messages": [
                {"role": "user", "content": "Say hello"}
            ],
            "maxTokens": 100
        }
    ]
    
    for i, payload in enumerate(payloads, 1):
        print(f"\n--- Trying Payload {i} ---")
        print(json.dumps(payload, indent=2))
        
        try:
            response = requests.post(
                'https://api.ntth.ai/v1/chat',
                json=payload,
                headers=headers,
                timeout=30
            )
            
            print(f"\nStatus Code: {response.status_code}")
            
            if response.status_code == 200:
                print("✓ SUCCESS!")
                print(f"Response: {response.json()}")
                return True
            else:
                print(f"✗ Failed")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"✗ Exception: {e}")
    
    return False

if __name__ == "__main__":
    test_api_directly()
