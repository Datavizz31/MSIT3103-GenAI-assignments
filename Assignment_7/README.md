# Simple Generative AI Model Deployment

This project demonstrates a simple generative AI model deployment using GPT-2 that runs on CPU.

## Project Structure

```
Cloud_deploy/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker containerization
├── gen_ai_model.py       # Main generative AI model
├── api_server.py         # FastAPI REST API server
├── test_model.py         # Test script
├── frontend/              # React + Vite frontend
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
└── models/                # Model directory
    └── __init__.py
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip
- AWS CLI configured (for cloud deployment)
- Docker (for containerization bonus)

### Installation
```bash
pip install -r requirements.txt
```

### Local Deployment
```bash
cd local_deployment
uvicorn main:app --reload
```

### Cloud Deployment
```bash
cd cloud_deployment
python sagemaker_deploy.py
```

### Docker Deployment (Bonus)
```bash
cd docker
docker-compose up --build
```

## Setup and Usage Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Model Directly

Run the simple generative AI model:

```bash
python simple_gen_ai.py
```

This will:
- Load the GPT-2 model
- Run test prompts
- Show generation times and results

### 3. Test with the Test Script

```bash
python test_model.py
```

### 4. Start the FastAPI Server

```bash
python api_server.py
```

The server will start at `http://localhost:8000`

### 5. Test the API

In another terminal:

```bash
python test_model.py api
```

Or visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

- `GET /` - Serve the chat UI
- `GET /health` - Health check
- `POST /generate/` - Generate text from prompt
- `GET /docs` - API documentation

### Example API Usage

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "The future of AI is", "max_length": 80, "temperature": 0.8}'
```

# Local Deployment: FastAPI REST API for Generative AI Model

## Setup Instructions

1. **Install dependencies:**
   ```sh
   pip install fastapi uvicorn transformers torch
   ```
   (If you haven't already, also install requirements from `requirements.txt`)

2. **Start the API server:**
   ```sh
   uvicorn api_server:app --reload
   ```
   The API will be available at http://127.0.0.1:8000

## API Endpoints

### 1. Health Check
- **GET** `/health`
- **Response:** `{ "status": "ok" }`

### 2. Text Generation
- **POST** `/generate/`
- **Request Body (JSON):**
  ```json
  {
    "prompt": "Your prompt here",
    "max_length": 80,
    "temperature": 0.5,
    "num_return_sequences": 1
  }
  ```
- **Response:**
  ```json
  {
    "outputs": ["Generated text..."],
    "latency_seconds": 1.23
  }
  ```

## Local Performance Benchmarking
- The `/generate/` endpoint returns the latency (in seconds) for each request.
- For throughput, you can send multiple requests and measure responses per second using tools like `ab` (ApacheBench) or `wrk`.

---

This setup provides a simple, production-ready local REST API for your generative AI model using FastAPI.

## Model Details

- **Model**: GPT-2 (small, ~124M parameters)
- **Device**: CPU only (no GPU required)
- **Framework**: PyTorch + Hugging Face Transformers
- **Memory**: ~500MB RAM required
- **Inference Time**: ~2-5 seconds per generation on typical CPU

---

## 🐳 Docker Deployment

### Quick Start with Docker

Build and run the complete application (frontend + backend) in Docker:

```bash
# Build the Docker image
docker build -t gen-ai-app .

# Run the container
docker run -p 8000:8000 gen-ai-app
```

Then access the application at: `http://localhost:8000`

### What's Included

The Docker image contains:
- ✅ React frontend (built with Vite)
- ✅ FastAPI backend server
- ✅ GPT-2 generative AI model
- ✅ All dependencies (PyTorch, Transformers, FastAPI, etc.)

### Image Details

- **Base Images**: Node 18 (for build) + Python 3.11 (runtime)
- **Size**: ~2.9 GB (includes PyTorch and model weights)
- **Build Time**: 5-10 minutes (initial build)
- **Port**: 8000

### Detailed Docker Documentation

For comprehensive Docker instructions including:
- Advanced running options (detached mode, volumes, environment variables)
- API endpoint examples
- Troubleshooting guide
- Production deployment considerations
- Performance optimization tips

👉 **See [DOCKER.md](./DOCKER.md) for complete documentation**

### Common Docker Commands

```bash
# Build
docker build -t gen-ai-app .

# Run (foreground)
docker run -p 8000:8000 gen-ai-app

# Run (background)
docker run -d -p 8000:8000 --name my-app gen-ai-app

# View logs
docker logs -f <container_id>

# Stop
docker stop <container_id>

# Remove
docker rm <container_id>
```

---

## Next Steps

Once this basic setup is working, we can proceed to:

1. **AWS SageMaker Deployment** - Deploy the model to AWS cloud
2. **Docker Containerization** - Package the application in Docker
3. **Performance Optimization** - Improve inference speed and scalability

## Troubleshooting

### Common Issues

1. **Model download fails**: Check internet connection, the model will be downloaded automatically on first run
2. **Out of memory**: Reduce `max_length` parameter in generation requests
3. **Slow inference**: Normal for CPU inference, consider smaller prompts

### Requirements Issues

If you get dependency conflicts, try:

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Assignment Components

### 1. Cloud Deployment (AWS SageMaker)
- Model packaging and deployment
- Endpoint configuration
- Inference testing
- Performance analysis

### 2. Local Deployment (FastAPI)
- REST API implementation
- Request/response handling
- Local performance benchmarks

### 3. Docker Containerization (Bonus)
- Dockerfile creation
- Container optimization
- Multi-stage builds

## Documentation
Detailed documentation for each deployment method can be found in the `docs/` directory.

## Performance Comparison
Performance metrics and analysis comparing cloud vs local deployment are documented in `docs/performance_analysis.md`.

### Docker Deployment

To containerize the application using Docker:

1. Ensure Docker is installed and running.

2. Build the Docker image:

```bash
docker build -t gen-ai-app .
```

3. Run the container:

```bash
docker run -p 8000:8000 gen-ai-app
```

The application will be available at `http://localhost:8000`, serving both the API and the frontend.

## Local Deployment

To run the application locally:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. For the frontend (optional, since backend serves it):

```bash
cd frontend
npm install
npm run dev
```

3. Start the FastAPI server:

```bash
python api_server.py
```

The application will be available at `http://localhost:8000`, serving both the API and the chat UI.
