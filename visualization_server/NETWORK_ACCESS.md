# 네트워크 접근 가이드

다른 컴퓨터에서 시각화 서버에 접근하는 방법입니다.

## 서버 설정

서버는 기본적으로 `0.0.0.0`으로 바인딩되어 모든 네트워크 인터페이스에서 접근 가능합니다.

## 접근 방법

### 1. 같은 네트워크에 있는 경우

1. **서버 IP 주소 확인**:
   ```bash
   # Linux/WSL
   hostname -I
   # 또는
   ip addr show | grep "inet "
   
   # Windows에서 WSL IP 확인
   wsl hostname -I
   ```

2. **서버 시작**:
   ```bash
   ./start_server.sh
   # 또는
   python visualization_server/app.py --port 5000 --host 0.0.0.0
   ```

3. **다른 컴퓨터에서 접속**:
   ```
   http://<서버_IP>:5000
   ```
   예: `http://192.168.1.100:5000`

### 2. 원격 접근 (SSH 터널링)

SSH를 통해 접근하는 방법:

```bash
# 로컬에서 SSH 터널 생성
ssh -L 5000:localhost:5000 user@server_ip

# 그 다음 브라우저에서
http://localhost:5000
```

### 3. 방화벽 설정

방화벽이 활성화되어 있으면 포트를 열어야 합니다:

**Linux (ufw)**:
```bash
sudo ufw allow 5000/tcp
sudo ufw status
```

**Windows Firewall**:
- 제어판 > Windows 방화벽 > 고급 설정
- 인바운드 규칙 > 새 규칙
- 포트 선택 > TCP > 특정 로컬 포트: 5000
- 연결 허용

**WSL2**:
WSL2는 Windows 방화벽을 통해야 하므로, Windows 방화벽에서 포트를 열어야 합니다.

## 보안 주의사항

⚠️ **중요**: 프로덕션 환경에서는 보안을 고려해야 합니다:

1. **인증 추가**: Flask-Login 등으로 인증 구현
2. **HTTPS 사용**: SSL/TLS 인증서 설정
3. **방화벽 제한**: 특정 IP만 접근 허용
4. **포트 변경**: 기본 포트 대신 다른 포트 사용

## 문제 해결

### 연결이 안 될 때:

1. **서버가 실행 중인지 확인**:
   ```bash
   netstat -tuln | grep 5000
   # 또는
   lsof -i :5000
   ```

2. **방화벽 확인**:
   ```bash
   sudo ufw status
   ```

3. **네트워크 확인**:
   ```bash
   ping <서버_IP>
   ```

4. **서버 로그 확인**: 서버 콘솔에서 접속 시도가 보이는지 확인

## 예시

같은 네트워크의 다른 컴퓨터에서 접근:

```bash
# 서버 컴퓨터에서
cd VAE-Planner
./visualization_server/start_server.sh

# 서버 IP가 192.168.1.100인 경우
# 다른 컴퓨터의 브라우저에서
http://192.168.1.100:5000
```
