from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

with open("prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

question = "How is the city using facial recognition technology to keep our children safe, and what measures are in place to protect their privacy?"

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]
)

answer = response.choices[0].message.content.strip()

print(answer)