FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot code into the container
COPY kindler.py .
COPY Bookerly.ttf .
COPY Aileron-Bold.otf .
COPY logo.jpg .

CMD [ "python", "kindler.py" ]
