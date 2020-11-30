# Base Image	
FROM python:3.9.0

# create and use folder for the image
WORKDIR /code

# Port
ENV PORT 5000

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Get files to create image and indicate where to put them
COPY app.py /code/app.py
COPY . /code

# Create an unprivileged user
RUN useradd --system --user-group --shell /sbin/nologin services

# Switch to the unprivileged user
USER services

# Run image as a container
CMD ["python", "app.py"]