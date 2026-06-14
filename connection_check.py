from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
with open("prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()
print(system_prompt[:80])

api_key = os.getenv("OPENROUTER_API_KEY")

print(api_key[:15])

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b:free",

    messages=[
        {
            "role": "user",
            "content": "Write one short question about smart-city cameras."
        }
    ],
)

print(response.choices[0].message.content)