# Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./app ./app

# Set the command to run your bot
CMD ["python", "app/main.py"]
