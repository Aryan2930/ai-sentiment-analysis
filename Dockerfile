FROM python:3.10-slim

WORKDIR /app

# Install system deps (needed for gradio/pydub)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["python", "app.py"]
