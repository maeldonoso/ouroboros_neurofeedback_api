# Pull base image. 
FROM python:3.11

# Set environment variables. 
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory. 
WORKDIR /code

# Install dependencies. 
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy project. 
COPY . .