# Use the official TensorFlow image as the base image
FROM tensorflow/tensorflow:2.12.0

# Copy the requirements.txt file to the container
COPY requirements.txt /

# Install the Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the source code to the container
COPY . /

# Set the environment variables
ENV PYTHONUNBUFFERED=1

# RUN apt-get update && apt-get install -y libgl1-mesa-glx
# Expose the port that Kserve will use to access the model (e.g., 8080)
EXPOSE 8080


# Run the command to start your application
CMD [ "python3", "inference.py" ]

