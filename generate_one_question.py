from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

with open("question_prompt.txt", "r", encoding="utf-8") as file:
    question_prompt = file.read()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

topic = "facial recognition"
user_style = "concerned parent"

user_prompt = f"""
Topic: {topic}
User style: {user_style}

Generate the question now.
"""

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": question_prompt},
        {"role": "user", "content": user_prompt}
    ],
)

question = response.choices[0].message.content.strip()

print(question)