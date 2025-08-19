# GPU VM Setup Guide - Vast.ai Deployment

## Complete Step-by-Step Process for RAG Evaluation on GPU VM

### Prerequisites
- Vast.ai account with credits
- SSH key pair on local machine
- RAG evaluation project ready

---

## Phase 1: Vast.ai Account Setup

### 1. Create Account & Add SSH Key
```bash
# Generate SSH key if needed
ssh-keygen -t ed25519 -C "your-email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub
```

**Vast.ai Console:**
- Add SSH public key to account settings
- Add credits for VM rental

### 2. VM Template Configuration

**Use: Ubuntu 22.04 CLI (VM)**

**Template Modifications:**
```bash
# Docker Options (add to template)
-p 741641:741641/udp -p 3000:3000 -p 4000:4000 -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 11434:11434

# Environment Variables (add to template)
NVIDIA_VISIBLE_DEVICES=all
DOCKER_BUILDKIT=1
```

**Enhanced On-start Script:**
```bash
#!/bin/bash
echo "Starting RAG Evaluation Environment Setup..."
cd /root

# Update system
apt-get update

# Install docker-compose if not present
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Install essentials
apt-get install -y git curl htop tree

echo "System ready for RAG deployment"
```

---

## Phase 2: VM Deployment

### 3. Rent Instance
**Recommended Specs:**
- **GPU**: RTX 3060+ (12GB+ VRAM)
- **RAM**: 24GB+
- **Storage**: 130GB+
- **Price**: $0.005-0.50/hr depending on GPU

### 4. Connect & Verify
```bash
# Connect to VM (get details from Vast.ai console)
ssh -p PORT root@IP_ADDRESS

# Verify GPU access
nvidia-smi

# Verify Docker
docker --version
docker-compose --version
```

---

## Phase 3: Project Deployment

### 5. Upload Project
```bash
# From local machine
cd /Users/Malik/code/malikbou/ucl/thesis/rag-eval
scp -P PORT -r ./rag-deployment/* root@IP_ADDRESS:/root/rag-deployment/
```

### 6. Create Environment File
```bash
# On VM
cd /root/rag-deployment

cat > .env << 'EOF'
# Environment variables for RAG deployment
OPENAI_API_KEY=dummy-key-for-litellm
LITELLM_MASTER_KEY=your-master-key
NVIDIA_VISIBLE_DEVICES=all
DOCKER_BUILDKIT=1
EOF
```

### 7. Fix Docker Compose Configuration
```bash
# Fix docker-compose.vm.yml to include volume definition
cat > docker-compose.vm.yml << 'EOF'
# This file is an add-on for the main docker-compose.yml.
# It should ONLY be used in a production environment with an NVIDIA GPU.

version: '3.9'

services:
  ollama:
    image: ollama/ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    container_name: ollama

volumes:
  ollama_data:
EOF
```

---

## Phase 4: Service Deployment

### 8. Deploy Docker Stack
```bash
# Validate configuration
docker-compose -f docker-compose.yml -f docker-compose.vm.yml config --quiet

# Start all services
docker-compose -f docker-compose.yml -f docker-compose.vm.yml up -d

# Check service status
docker-compose -f docker-compose.yml -f docker-compose.vm.yml ps
```

### 9. Load Ollama Models
```bash
# Pull required models for benchmarking
docker exec ollama ollama pull nemotron-mini:4b
docker exec ollama ollama pull llama3.2:3b
docker exec ollama ollama pull gemma3:4b

# Verify models loaded
docker exec ollama ollama list
```

---

## Phase 5: Enhanced Benchmarking

### 10. Run Extended Concurrency Tests
```bash
# Run comprehensive benchmark with GPU acceleration
docker exec ollama-benchmark python src/ollama_benchmark.py \
  --questions 10 \
  --concurrency-range "1,2,4,8,16,32,64,128" \
  --generate-charts

# Check results
ls -la results/ollama_benchmarks/
```

### 11. Download Results
```bash
# From local machine - download generated charts and data
scp -P PORT -r root@IP_ADDRESS:/root/rag-deployment/results/ollama_benchmarks/ ./results/
```

---

## Key Improvements to Make in Local Scripts

### 1. Enhanced Docker Compose Volume Fix
**File: `docker-compose.vm.yml`**
```yaml
# Always include volumes section
volumes:
  ollama_data:
```

### 2. Environment File Template
**File: `.env.template`**
```bash
# Copy this to .env and fill in values
OPENAI_API_KEY=dummy-key-for-litellm
LITELLM_MASTER_KEY=your-master-key
NVIDIA_VISIBLE_DEVICES=all
DOCKER_BUILDKIT=1
```

### 3. VM Deployment Script
**File: `deploy-to-vm.sh`**
```bash
#!/bin/bash
set -e

VM_IP="$1"
VM_PORT="$2"

if [ -z "$VM_IP" ] || [ -z "$VM_PORT" ]; then
    echo "Usage: $0 <VM_IP> <VM_PORT>"
    exit 1
fi

echo "Deploying to VM at $VM_IP:$VM_PORT"

# Upload project files
scp -P $VM_PORT -r ./rag-deployment/* root@$VM_IP:/root/rag-deployment/

# Connect and deploy
ssh -p $VM_PORT root@$VM_IP << 'EOF'
cd /root/rag-deployment

# Create .env if not exists
if [ ! -f .env ]; then
    cp .env.template .env
fi

# Deploy services
docker-compose -f docker-compose.yml -f docker-compose.vm.yml up -d

# Load models
docker exec ollama ollama pull nemotron-mini:4b &
docker exec ollama ollama pull llama3.2:3b &
docker exec ollama ollama pull gemma3:4b &
wait

echo "Deployment complete!"
EOF
```

### 4. Benchmark Runner Script
**File: `run-gpu-benchmark.sh`**
```bash
#!/bin/bash
# Run extended benchmarks on GPU VM

QUESTIONS=${1:-10}
CONCURRENCY=${2:-"1,2,4,8,16,32,64,128"}

echo "Running GPU benchmark with $QUESTIONS questions, concurrency: $CONCURRENCY"

docker exec ollama-benchmark python src/ollama_benchmark.py \
  --questions $QUESTIONS \
  --concurrency-range "$CONCURRENCY" \
  --generate-charts

echo "Benchmark complete! Results in results/ollama_benchmarks/"
```

---

## Service Access Methods

### Option 1: Direct Port Access
- OpenWebUI: `http://VM_IP:3000`
- LiteLLM: `http://VM_IP:4000`
- RAG API: `http://VM_IP:8000`

### Option 2: SSH Port Forwarding (Secure)
```bash
ssh -p VM_PORT -L 3000:localhost:3000 -L 4000:localhost:4000 -L 8000:localhost:8000 root@VM_IP
# Access via localhost:3000, localhost:4000, etc.
```

### Option 3: Tailscale (Team Access)
```bash
# On VM
sudo tailscale up
# Access via tailscale private IP
```

---

## Cost Optimization Tips

1. **Stop instance when not in use**
```bash
# From VM
vastai stop instance $CONTAINER_ID
```

2. **Use spot instances** for longer runs
3. **Download results before stopping** instance
4. **Use cheaper GPUs** for development, upgrade for final benchmarks

---

## Troubleshooting

### Common Issues:
1. **SSH connection failed**: Wait 30+ minutes for VM boot
2. **GPU not detected**: Verify NVIDIA container toolkit
3. **Port access denied**: Check VM firewall/vast.ai port configuration
4. **Models won't load**: Check disk space and GPU memory

### Debug Commands:
```bash
# Check GPU in container
docker exec ollama nvidia-smi

# Check service logs
docker-compose logs ollama

# Check disk space
df -h

# Check container status
docker-compose ps
```

---

This setup provides professional-grade GPU acceleration for your RAG evaluation thesis, enabling concurrency testing impossible on local hardware while maintaining cost efficiency through pay-per-use cloud resources.
