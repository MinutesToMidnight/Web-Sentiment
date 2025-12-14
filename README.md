# Web-Sentiment

It is a small containerized FastAPI application for sentiment classification, built using free models via OpenRouter.

---

## Prerequisites
- **Docker** and **Docker Compose** installed.  
- **OpenRouter** account to obtain an API key.  
- (Optional) **GitHub** account to run CI and store secrets.

---

## Quick start

### 1. Clone the repository
```bash
# bash
git clone https://github.com/MinutesToMidnight/Web-Sentiment.git
cd Web-Sentiment
```
Or download zip-file of the project and extract it to the desired folder.

### 2. Create local environment file
Create a .env file in the repository root (do not commit it). See Configuration and OpenRouter API key for required variables.
In order to get an API key, sign up on https://openrouter.ai, hover over your profile, select "Keys" tab, create a key, save it somewhere to not lost it and paste to the .env file, which you should create in "app" folder.
```.env
OPENROUTER_KEY=your_api_key
```
Add env_file to the service in docker-compose.yml:
```yaml
# docker-compose.yml
services:
  web:
    build: .
    env_file:
      - .env
    ports:
      - "8000:8000"
```

### 3. Build and run
Navigate using bash to the folder where the project is installed and execute the following.
```bash
# bash
docker compose build --no-cache
docker compose up -d
```

### 4. Check service
```bash
# bash
curl -v http://127.0.0.1:8000/health
# or open http://127.0.0.1:8000/ in a browser
```

### 5. Stop and remove containers
```bash
docker compose down
```
