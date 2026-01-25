import React, { Component } from "react";
import LatentSpacePlot from "./components/LatentSpacePlot";
import TrajectoryCanvas from "./components/TrajectoryCanvas";
import "./App.css";

// 개발 모드에서는 환경 변수 사용, 프로덕션에서는 같은 서버에서 제공하므로 상대 경로
const API_BASE_URL = process.env.REACT_APP_API_URL || "";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      datasetInfo: null,
      latentSpaceData: null,
      selectedTrajectory: null,  // hover된 trajectory
      selectedLatent: null,  // hover된 latent z
      trajectoryBounds: null,  // 모든 trajectory의 전체 범위
      loading: false,
      error: null,
      method: 'tsne'  // 기본값: t-SNE
    };
    
    this.handleLatentHover = this.handleLatentHover.bind(this);
    this.handleMethodChange = this.handleMethodChange.bind(this);
  }

  componentDidMount() {
    this.loadDatasetInfo();
    this.loadLatentSpace(this.state.method);
  }
  
  handleMethodChange(method) {
    this.setState({ method, latentSpaceData: null, selectedTrajectory: null, selectedLatent: null });
    this.loadLatentSpace(method);
  }

  async loadDatasetInfo() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/dataset-info`);
      const data = await response.json();
      this.setState({ datasetInfo: data });
    } catch (error) {
      this.setState({ error: `Failed to load dataset info: ${error.message}` });
    }
  }

  async loadLatentSpace(method = 'tsne') {
    this.setState({ loading: true, error: null });
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/latent-space`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          method: method,  // 'tsne' or 'umap'
          max_samples: 10000  // 샘플 수를 10,000개로 고정
        })
      });
      
      // 응답 상태 확인
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: `HTTP ${response.status}: ${response.statusText}` }));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // 에러 응답 확인
      if (data.error) {
        throw new Error(data.error);
      }
      
      // 데이터 유효성 확인
      if (!data.projected_points || !Array.isArray(data.projected_points)) {
        throw new Error('Invalid response format: missing projected_points');
      }
      
      console.log(`Loaded ${data.num_samples || data.projected_points.length} samples (total dataset: ${data.total_dataset_size || 'unknown'})`);
      
      // 모든 trajectory의 전체 범위 계산
      let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
      data.projected_points.forEach(point => {
        if (point.trajectory && point.trajectory.length > 0) {
          point.trajectory.forEach(trajPoint => {
            xMin = Math.min(xMin, trajPoint[0]);
            xMax = Math.max(xMax, trajPoint[0]);
            yMin = Math.min(yMin, trajPoint[1]);
            yMax = Math.max(yMax, trajPoint[1]);
          });
        }
      });
      
      // 범위가 없으면 기본 범위 설정
      if (xMin === Infinity) {
        const defaultRange = 100;
        xMin = -defaultRange;
        xMax = defaultRange;
        yMin = -defaultRange;
        yMax = defaultRange;
      }
      
      // 약간의 padding 추가
      const padding = Math.max((xMax - xMin) * 0.1, (yMax - yMin) * 0.1, 10);
      const bounds = {
        xMin: xMin - padding,
        xMax: xMax + padding,
        yMin: yMin - padding,
        yMax: yMax + padding
      };
      
      this.setState({ 
        latentSpaceData: data, 
        trajectoryBounds: bounds,
        loading: false 
      });
    } catch (error) {
      console.error('Error loading latent space:', error);
      this.setState({ 
        error: `Failed to load latent space: ${error.message}`,
        loading: false 
      });
    }
  }

  handleLatentHover(latent, trajectory) {
    // Latent z에 hover하면 해당 trajectory 표시
    this.setState({
      selectedTrajectory: trajectory,
      selectedLatent: latent
    });
  }

  render() {
    const { 
      datasetInfo,
      latentSpaceData, 
      selectedTrajectory,
      selectedLatent,
      trajectoryBounds,
      loading,
      error 
    } = this.state;

    return (
      <div className="App">
        <header>
          <div className="header-left">
            <div className="header-logo">VAE</div>
            <div className="header-titles">
              <h1>VAE-Planner Latent Space Explorer</h1>
              <span className="header-subtitle">Trajectory Latent Space Visualization</span>
            </div>
          </div>
          <div className="header-right">
            {datasetInfo && (
              <div className="dataset-info">
                <span>Dataset: {datasetInfo.num_samples.toLocaleString()} trajectories</span>
                <span>Latent Dim: {datasetInfo.latent_dim}</span>
              </div>
            )}
            <div className="method-selector">
              <label>Method: </label>
              <button 
                className={this.state.method === 'tsne' ? 'active' : ''}
                onClick={() => this.handleMethodChange('tsne')}
                disabled={this.state.loading}
              >
                t-SNE
              </button>
              <button 
                className={this.state.method === 'umap' ? 'active' : ''}
                onClick={() => this.handleMethodChange('umap')}
                disabled={this.state.loading}
              >
                UMAP
              </button>
            </div>
          </div>
        </header>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
        
        {loading && (
          <div className="loading-overlay">
            <div className="spinner">
              <span>Loading latent space...</span>
            </div>
          </div>
        )}

        <main className="main-content">
          {!latentSpaceData ? (
            <div className="empty-state">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M9 17.25v1.007a3 3 0 01-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0115 18.257V17.25m6-12V15a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 15V5.25m18 0A2.25 2.25 0 0018.75 3H5.25A2.25 2.25 0 003 5.25m18 0V12a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 12V5.25" />
              </svg>
              <h3>Latent space를 로딩 중입니다...</h3>
              <p>데이터셋의 trajectory들이 latent space에 projection되고 있습니다.</p>
            </div>
          ) : (
            <>
              <div className="visualization-grid">
                <section className="card">
                  <h2 className="card-title">Latent Space (2D Projection)</h2>
                  <p className="card-description">
                    마우스를 움직여 latent z에 hover하면 해당 trajectory가 오른쪽에 표시됩니다.
                    {latentSpaceData && (
                      <>
                        <span style={{marginLeft: '10px', color: '#64748b', fontSize: '0.85em'}}>
                          Method: {latentSpaceData.method.toUpperCase()}
                        </span>
                        {latentSpaceData.latent_stats && (
                          <span style={{marginLeft: '10px', fontSize: '0.85em'}}>
                            {latentSpaceData.latent_stats.posterior_collapse_warning ? (
                              <span style={{color: '#ef4444', fontWeight: 'bold'}}>
                                ⚠️ Posterior Collapse 가능성 (std: {latentSpaceData.latent_stats.mean_std.toFixed(4)})
                              </span>
                            ) : (
                              <span style={{color: '#10b981'}}>
                                ✓ Latent std: {latentSpaceData.latent_stats.mean_std.toFixed(4)}
                              </span>
                            )}
                          </span>
                        )}
                      </>
                    )}
                  </p>
                  {latentSpaceData && (
                    <LatentSpacePlot
                      data={latentSpaceData.projected_points}
                      onHover={this.handleLatentHover}
                    />
                  )}
                </section>

                <section className="card">
                  <h2 className="card-title">Trajectory Visualization</h2>
                  {selectedTrajectory ? (
                    <>
                      <p className="card-description">
                        Hover된 latent z에 매칭되는 원본 입력 경로입니다.
                      </p>
                      <TrajectoryCanvas
                        trajectory={selectedTrajectory}
                        bounds={trajectoryBounds}
                      />
                    </>
                  ) : (
                    <div className="empty-trajectory">
                      <p>Latent space에서 마우스를 움직여 trajectory를 선택하세요.</p>
                    </div>
                  )}
                </section>
              </div>
            </>
          )}
        </main>
      </div>
    );
  }
}

export default App;
