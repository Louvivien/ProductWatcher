# Use an official Python runtime as a parent image
FROM python:3.10 as builder


# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Add a non-root user and switch to it
RUN adduser --disabled-password --gecos '' app
USER app

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Make port 8000 available to the world outside this container
EXPOSE 8080

# Run gunicorn when the container launches
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8080"]



