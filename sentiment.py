import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_sentiment(chat_history):
    if not chat_history:
        return 5

    history_text = "\n".join(
        [f"{sender}: {message}" for sender, message, _ in chat_history]
    )

    prompt = f"""
You are analyzing a support chat conversation between a user and an AI assistant.
Based on the entire conversation, rate how satisfied the user is with the support they are receiving.

Give a score from 1 to 10 where:
1 = Very unsatisfied, frustrated, problem not solved
5 = Neutral, partially helped
10 = Very satisfied, problem fully resolved

Conversation:
{history_text}

Reply with ONLY a single number between 1 and 10. Nothing else.
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    try:
        score = int(response.text.strip())
        if score < 1:
            score = 1
        if score > 10:
            score = 10
        return score
    except:
        return 5