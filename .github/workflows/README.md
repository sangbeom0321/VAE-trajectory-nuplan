# GitHub Pages Deployment

This directory contains GitHub Actions workflow for deploying the VAE-Planner visualization server frontend to GitHub Pages.

## Overview

The workflow automatically builds and deploys the React frontend to GitHub Pages when you push to the `main` or `master` branch. The Flask backend API needs to be hosted separately on a cloud service.

## Workflow Details

The workflow (`deploy-pages.yml`) performs the following steps:

1. **Checkout**: Checks out the repository code
2. **Setup Node.js**: Sets up Node.js 18 with npm caching
3. **Install Dependencies**: Installs npm dependencies from `visualization_server/package-lock.json`
4. **Build React App**: Builds the React frontend with the backend API URL from secrets
5. **Setup Pages**: Configures GitHub Pages
6. **Upload Artifact**: Uploads the built frontend files
7. **Deploy**: Deploys to GitHub Pages

## Backend Hosting Options

Since GitHub Pages only serves static files, you need to host the Flask backend separately. Recommended options:

1. **Render** (Recommended): https://render.com
   - Free tier available
   - Easy deployment from GitHub
   - Automatic HTTPS

2. **Railway**: https://railway.app
   - Simple deployment process
   - Good free tier

3. **Heroku**: https://heroku.com
   - Well-established platform
   - Requires credit card for free tier

4. **Fly.io**: https://fly.io
   - Good performance
   - Free tier available

## Setup Instructions

### 1. Enable GitHub Pages

1. Go to your repository Settings → Pages
2. Select "GitHub Actions" as the source
3. The workflow will automatically deploy when triggered

### 2. Deploy Backend

Deploy the Flask backend (`visualization_server/app.py`) to one of the hosting services above. Make sure to:

- Set environment variables for data paths
- Configure CORS to allow requests from your GitHub Pages domain
- Note the backend URL (e.g., `https://your-backend.herokuapp.com`)

### 3. Configure Frontend-Backend Connection

1. Go to your repository Settings → Secrets and variables → Actions
2. Add a new repository secret:
   - Name: `REACT_APP_API_URL`
   - Value: Your backend API URL (e.g., `https://your-backend.herokuapp.com`)

### 4. Deploy

1. Push to `main` or `master` branch, or
2. Manually trigger the workflow:
   - Go to Actions tab
   - Select "Deploy to GitHub Pages" workflow
   - Click "Run workflow"

The frontend will be automatically built and deployed to GitHub Pages.

## Manual Workflow Trigger

You can also manually trigger the deployment:

1. Go to the Actions tab in your repository
2. Select "Deploy to GitHub Pages" workflow
3. Click "Run workflow" button
4. Select the branch and click "Run workflow"

## Troubleshooting

### Build Fails

- Check that `visualization_server/package.json` and `package-lock.json` are present
- Verify Node.js version compatibility (requires Node.js 14+)
- Check workflow logs for specific error messages

### Frontend Can't Connect to Backend

- Verify `REACT_APP_API_URL` secret is set correctly
- Check backend CORS configuration allows your GitHub Pages domain
- Ensure backend is running and accessible

### Pages Not Updating

- Wait a few minutes for GitHub Pages to update
- Check Actions tab to ensure workflow completed successfully
- Clear browser cache and hard refresh (Ctrl+F5 or Cmd+Shift+R)

## Repository Structure

```
.github/
└── workflows/
    ├── deploy-pages.yml    # GitHub Actions workflow
    └── README.md           # This file
```

## Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [React Build Documentation](https://create-react-app.dev/docs/production-build/)
