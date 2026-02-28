import requests
import json
import os

def test_tts_endpoint():
    url = "http://localhost:8000/text-to-speech"
    
    # User provided sample data
    # Note: Using the exact text provided
    complex_script = [
        {
            'Host': "That's fascinating! So, with all these incredible advancements, what do you think will be the biggest challenge we face in actually establishing a permanent human presence on Mars? \n\n", 
            'Guest': "The biggest challenge won't be the technology, but the human element.  Sustaining a complex society in a harsh, isolated environment will require immense psychological resilience and the ability to resolve conflicts effectively, lessons we're still learning here on Earth.  Think of it like a giant, interplanetary social experiment: can we truly work together and thrive in a completely new world?  \n\n"
        }, 
        {
            'Host': "That's a really thought-provoking point!  So, how do you envision us preparing for those social and psychological challenges before we even set foot on Mars?  Will it take a whole new approach to education, leadership training, or even our understanding of human nature? \n\n", 
            'Guest': "Absolutely. We need to shift from solely focusing on technical expertise to nurturing  interpersonal skills, conflict resolution, and adaptability in our astronaut candidates. Think of it like training for an Olympic team, but instead of physical prowess, we're focusing on emotional intelligence and collaborative problem-solving  - essential traits for thriving in a confined, Martian habitat. \n\n"
        }
    ]
    
    payload = {
        "script": complex_script,
        "language": "English"
    }
    
    print(f"Sending request to {url}...")
    print(f"Payload script length: {len(payload['script'])} turns")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        print("Response received:")
        print(json.dumps(data, indent=2))
        
        # Verify file exists
        audio_file = data.get("audio_file")
        if audio_file and os.path.exists(audio_file):
            print(f"Combined audio exists: {audio_file} ({os.path.getsize(audio_file)} bytes)")
        else:
            print(f"Combined audio missing: {audio_file}")
                
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response:
             print(e.response.text)

if __name__ == "__main__":
    test_tts_endpoint()
