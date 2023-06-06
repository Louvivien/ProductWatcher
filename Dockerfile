# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add everything in the current directory to our image, in /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Firefox
RUN apt-get update && apt-get install -y firefox-esr

# Install geckodriver required by Selenium
RUN apt-get install wget && \
    wget https://github.com/mozilla/geckodriver/releases/download/v0.29.1/geckodriver-v0.29.1-linux64.tar.gz && \
    tar -xvzf geckodriver-v0.29.1-linux64.tar.gz && \
    chmod +x geckodriver && \
    mv geckodriver /usr/local/bin/

# Run app.py when the container launches
CMD ["python", "app.py"]
