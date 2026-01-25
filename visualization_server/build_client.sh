#!/bin/bash

# React 클라이언트 빌드 스크립트

echo "Building React client..."

# Node.js 의존성 설치
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# React 앱 빌드
echo "Building React app..."
npm run build

# build 폴더를 client_build로 복사
if [ -d "build" ]; then
    echo "Copying build files to client_build..."
    rm -rf client_build
    cp -r build client_build
    echo "Build complete! Client files are in client_build/"
else
    echo "Error: Build folder not found!"
    exit 1
fi
