# Use an official Python runtime as a parent image
FROM python:3.10-slim as builder

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


# Copy the dependencies from the builder stage
COPY --from=builder /usr/local /usr/local


# Copy the application files (needed for gunicorn to find app:app)
COPY --from=builder /app /app

# Make sure the service listens on the port defined by the PORT environment variable
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app

