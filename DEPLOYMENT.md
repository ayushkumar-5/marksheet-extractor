# 🚀 Railway Deployment Guide

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **GitHub Repository**: Push your code to GitHub
3. **API Keys**: Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Step-by-Step Deployment

### 1. Prepare Your Repository

Make sure your repository contains:
- ✅ `main.py` - FastAPI application
- ✅ `requirements.txt` - Python dependencies
- ✅ `Dockerfile` - Container configuration
- ✅ `railway.json` - Railway configuration
- ✅ `.env.example` - Environment variables template

### 2. Push to GitHub

```bash
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

### 3. Deploy on Railway

1. **Go to Railway Dashboard**
   - Visit [railway.app](https://railway.app)
   - Click "New Project"

2. **Connect GitHub Repository**
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Select the main branch

3. **Configure Environment Variables**
   - Go to your project's "Variables" tab
   - Add the following variables:
   ```
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   PORT=8000
   LOG_LEVEL=INFO
   ```

4. **Deploy**
   - Railway will automatically detect the Dockerfile
   - Click "Deploy Now"
   - Wait for the build to complete

### 4. Access Your Application

Once deployed, Railway will provide:
- **Production URL**: `https://your-app-name.railway.app`
- **Demo UI**: `https://your-app-name.railway.app/demo`
- **API Docs**: `https://your-app-name.railway.app/docs`

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | ✅ Yes |
| `PORT` | Application port | ❌ No (default: 8000) |
| `LOG_LEVEL` | Logging level | ❌ No (default: INFO) |

## Troubleshooting

### Build Failures
- Check that all dependencies are in `requirements.txt`
- Ensure Dockerfile is properly configured
- Verify Python version in `runtime.txt`

### Runtime Errors
- Check Railway logs in the dashboard
- Verify environment variables are set correctly
- Ensure API keys are valid

### Health Check Failures
- Verify `/health` endpoint returns 200
- Check application startup logs
- Ensure port configuration is correct

## Monitoring

Railway provides:
- **Real-time logs** in the dashboard
- **Metrics** for CPU, memory, and network usage
- **Automatic restarts** on failures
- **Custom domains** (optional)

## Cost Optimization

- Railway offers **free tier** with limitations
- Monitor usage in the dashboard
- Consider upgrading for production use

## Security Notes

- Never commit `.env` files to Git
- Use Railway's environment variables for secrets
- Enable HTTPS (automatic on Railway)
- Consider adding authentication for production use

---

🎉 **Your AI Marksheet Extraction API is now live on Railway!**
