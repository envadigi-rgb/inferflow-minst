FROM mirror.gcr.io/library/python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model.pkl app.py .
ENV PORT=8080
EXPOSE 8080
CMD ["python", "app.py"]
