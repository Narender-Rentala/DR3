import os

api_key = input("Enter your OpenAI API Key: ")

with open(".env", "w") as f:
    f.write(f"OPENAI_API_KEY={api_key}\n")

print(".env file created successfully.")
