# GitHub Pages Deployment

This directory is automatically populated by GitHub Actions when deploying to GitHub Pages.

The React frontend is built and deployed here, while the Flask backend API needs to be hosted separately.

## Backend Hosting Options

1. **Render** (Recommended): https://render.com
2. **Railway**: https://railway.app
3. **Heroku**: https://heroku.com
4. **Fly.io**: https://fly.io

## Setup Instructions

1. Deploy the Flask backend to one of the hosting services above
2. Set the backend URL in GitHub repository secrets: `REACT_APP_API_URL`
3. Push to main/master branch to trigger automatic deployment
