################################################################################
# Multi-stage Dockerfile for Generative AI Application
# 
# This Dockerfile uses a multi-stage build approach to create an optimized
# production image that includes:
# - React frontend (built with Vite)
# - Python FastAPI backend
# - GPT-2 generative AI model
#
# Build Time: ~5-10 minutes (initial build)
# Image Size: ~2.9 GB (includes PyTorch and model weights)
################################################################################

# ============================================================================
# STAGE 1: Build the React Frontend
# ============================================================================
# This stage builds the React application using Node.js and Vite
# The built files will be copied to the final image
FROM node:18 AS frontend

WORKDIR /app/frontend

# Copy package configuration files
# package.json: Project metadata and dependencies
# package-lock.json: Locked dependency versions (if exists)
COPY frontend/package*.json ./

# Install all npm dependencies
# Using npm ci would be more reproducible in production
RUN npm install

# Copy the complete frontend source code
COPY frontend/ ./

# Build the production-optimized frontend bundle
# Output: frontend/dist/ (static files to be served)
RUN npm run build

# ============================================================================
# STAGE 2: Setup the Python Backend (Final Image)
# ============================================================================
# This stage creates the final image with Python backend and built frontend
FROM python:3.11

WORKDIR /app

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install Python Dependencies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
COPY requirements.txt ./

# Install Python dependencies without cache to reduce layer size
# Dependencies include:
# - FastAPI: Web framework for building the API
# - Uvicorn: ASGI server to run the application
# - PyTorch & Transformers: For running the GPT-2 model
# - Other utilities: pydantic, aiofiles, pytest, etc.
RUN pip install --no-cache-dir -r requirements.txt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copy Application Source Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copy the Python application files
COPY gen_ai_model.py ./
COPY api_server.py ./

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copy Built Frontend
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copy the compiled React app from the frontend build stage
# These static files will be served by FastAPI
COPY --from=frontend /app/frontend/dist ./frontend/dist

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Expose Port
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The FastAPI server will listen on port 8000
# Map this to your host port when running: -p 8000:8000 or -p <host_port>:8000
EXPOSE 8000

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Start the Application
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the FastAPI server using Uvicorn
# - app: The FastAPI instance defined in api_server.py
# - --host 0.0.0.0: Accept connections from all network interfaces
# - --port 8000: Listen on port 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
