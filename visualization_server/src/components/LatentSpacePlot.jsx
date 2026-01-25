import React, { Component } from "react";
import PropTypes from "prop-types";
import * as d3 from "d3";
import "./LatentSpacePlot.css";

class LatentSpacePlot extends Component {
  constructor(props) {
    super(props);
    this.renderChart = this.renderChart.bind(this);
    this.nodeRef = React.createRef();
  }

  componentDidMount() {
    this.renderChart();
  }

  componentDidUpdate() {
    this.renderChart();
  }

  renderChart() {
    const { data, onHover, onClick, method, mode } = this.props;
    if (!data || data.length === 0 || !this.nodeRef.current) return;

    const width = 800;
    const height = 700;  // 높이 증가: 600 → 700
    const margin = { top: 20, bottom: 150, left: 50, right: 20 };  // 하단 여백 더 증가

    let svg = d3.select(this.nodeRef.current);
    svg.selectAll("*").remove();

    svg.attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
       .attr("preserveAspectRatio", "xMidYMid meet");

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left}, ${margin.top})`);

    // 데이터 범위 계산
    const xExtent = d3.extent(data, d => d.x);
    const yExtent = d3.extent(data, d => d.y);

    const xScale = d3.scaleLinear()
      .domain(xExtent)
      .range([0, width])
      .nice();

    const yScale = d3.scaleLinear()
      .domain(yExtent)
      .range([height, 0])
      .nice();

    // 배경 사각형 (hover 영역)
    const hoverRect = g.append("rect")
      .attr("width", width)
      .attr("height", height)
      .style("fill", "transparent")
      .style("cursor", "crosshair");

    // 라벨별 색상 매핑 (확장된 분류)
    const labelColors = {
      'stop': '#ef4444',           // Red
      'straight': '#f59e0b',        // Orange - 직진
      'straight_sharp': '#fb923c',  // Orange-600 - 급커브 직진
      'straight_slow': '#fbbf24',   // Amber-400 - 느린 직진
      'left': '#3b82f6',            // Blue - 좌회전
      'left_sharp': '#2563eb',      // Blue-600 - 급커브 좌회전
      'left_slow': '#60a5fa',       // Blue-400 - 느린 좌회전
      'right': '#10b981',           // Green - 우회전
      'right_sharp': '#059669',     // Green-600 - 급커브 우회전
      'right_slow': '#34d399'       // Green-400 - 느린 우회전
    };
    
    const labelNames = {
      'stop': 'Stop',
      'straight': 'Straight',
      'straight_sharp': 'Straight (Sharp)',
      'straight_slow': 'Straight (Slow)',
      'left': 'Left Turn',
      'left_sharp': 'Left Turn (Sharp)',
      'left_slow': 'Left Turn (Slow)',
      'right': 'Right Turn',
      'right_sharp': 'Right Turn (Sharp)',
      'right_slow': 'Right Turn (Slow)'
    };
    
    // 라벨을 기본 카테고리로 그룹화 (색상은 유지하되 legend는 간소화)
    const getBaseLabel = (label) => {
      if (label === 'stop') return 'stop';
      if (label.startsWith('straight')) return 'straight';
      if (label.startsWith('left')) return 'left';
      if (label.startsWith('right')) return 'right';
      return label;
    };

    // 데이터 포인트 그리기 (라벨별 색상 적용)
    const points = g.selectAll("circle.point")
      .data(data)
      .enter()
      .append("circle")
      .attr("class", "point")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", 3)
      .attr("fill", d => labelColors[d.label] || "#6366f1")
      .attr("opacity", 0.7)
      .attr("stroke", d => labelColors[d.label] || "#4f46e5")
      .attr("stroke-width", 1);

    // 마우스 hover 이벤트 (Browse 모드에서만 활성화)
    let hoverTimeout = null;
    let hoveredPoint = null;
    
    if (mode !== 'generate' && onHover) {
      hoverRect.on("mousemove", async (event) => {
        const [mouseX, mouseY] = d3.pointer(event);
        const x = xScale.invert(mouseX);
        const y = yScale.invert(mouseY);
        
        if (hoverTimeout) {
          clearTimeout(hoverTimeout);
        }
        
        hoverTimeout = setTimeout(async () => {
          let minDist = Infinity;
          let nearestPoint = null;
          
          data.forEach(point => {
            const dist = Math.sqrt((point.x - x) ** 2 + (point.y - y) ** 2);
            if (dist < minDist) {
              minDist = dist;
              nearestPoint = point;
            }
          });
          
          if (nearestPoint && onHover) {
            // 가장 가까운 포인트 강조
            if (hoveredPoint) {
              // 이전 hover 포인트 스타일 복원
              points.filter(d => d === hoveredPoint)
                .attr("r", 3)
                .attr("opacity", 0.6);
            }
            
            // 새로운 hover 포인트 강조
            points.filter(d => d === nearestPoint)
              .attr("r", 6)
              .attr("opacity", 1.0);
            
            hoveredPoint = nearestPoint;
            
            // onHover 콜백 호출 (latent z, trajectory, 2D 좌표 전달)
            onHover(nearestPoint.latent, nearestPoint.trajectory, [nearestPoint.x, nearestPoint.y]);
          }
        }, 30);
      });

      hoverRect.on("mouseleave", () => {
        if (hoveredPoint) {
          points.filter(d => d === hoveredPoint)
            .attr("r", 3)
            .attr("opacity", 0.6);
          hoveredPoint = null;
        }
      });
    }
    
    // 클릭 이벤트: 빈 공간 클릭 시 해당 위치의 trajectory 생성
    hoverRect.on("click", async (event) => {
      const [mouseX, mouseY] = d3.pointer(event);
      const x = xScale.invert(mouseX);
      const y = yScale.invert(mouseY);
      
      // 클릭한 위치에 데이터 포인트가 있는지 확인
      let minDist = Infinity;
      let nearestPoint = null;
      
      data.forEach(point => {
        const dist = Math.sqrt((point.x - x) ** 2 + (point.y - y) ** 2);
        if (dist < minDist) {
          minDist = dist;
          nearestPoint = point;
        }
      });
      
      // 데이터 포인트와의 거리가 충분히 멀면 (빈 공간 클릭) 생성 요청
      const clickRadius = Math.min(
        (xExtent[1] - xExtent[0]) * 0.05,
        (yExtent[1] - yExtent[0]) * 0.05
      );
      
      if (minDist > clickRadius && onClick) {
        // 빈 공간 클릭: 서버에 trajectory 생성 요청
        onClick(x, y);
      } else if (nearestPoint && onClick) {
        // 데이터 포인트 클릭: 기존 trajectory 표시
        onClick(nearestPoint.x, nearestPoint.y, nearestPoint.trajectory, nearestPoint.latent);
      }
    });

    // 축 추가
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5);

    const gx = g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(xAxis);
    
    gx.selectAll("text").attr("fill", "#64748b").style("font-size", "12px");
    gx.selectAll("line").attr("stroke", "#e2e8f0");
    gx.select(".domain").attr("stroke", "#e2e8f0");

    const gy = g.append("g")
      .call(yAxis);
    
    gy.selectAll("text").attr("fill", "#64748b").style("font-size", "12px");
    gy.selectAll("line").attr("stroke", "#e2e8f0");
    gy.select(".domain").attr("stroke", "#e2e8f0");

    // 축 레이블 (method에 따라 동적으로 변경)
    const xLabel = method === 'pca' ? 'PC1' : method === 'tsne' ? 't-SNE 1' : 'UMAP 1';
    const yLabel = method === 'pca' ? 'PC2' : method === 'tsne' ? 't-SNE 2' : 'UMAP 2';
    
    g.append("text")
      .attr("x", width / 2)
      .attr("y", height + 35)
      .attr("fill", "#94a3b8")
      .style("text-anchor", "middle")
      .style("font-size", "12px")
      .style("font-weight", "500")
      .text(xLabel);

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -40)
      .attr("x", -height / 2)
      .attr("fill", "#94a3b8")
      .style("text-anchor", "middle")
      .style("font-size", "12px")
      .style("font-weight", "500")
      .text(yLabel);
    
    // Legend 추가 (plot 외부 하단으로 이동)
    const legend = g.append("g")
      .attr("transform", `translate(${width / 2 - 200}, ${height + 50})`);
    
    // 실제 데이터에서 사용된 라벨만 legend에 표시
    const usedLabels = [...new Set(data.map(d => d.label))].sort();
    const legendItems = usedLabels.map(label => ({
      label: label,
      name: labelNames[label] || label
    }));
    
    // Legend를 여러 열로 배치 (너무 많으면 2열로)
    const itemsPerColumn = Math.ceil(legendItems.length / 2);
    const legendWidth = 200;
    const itemHeight = 20;
    const legendHeight = itemsPerColumn * itemHeight + 25;
    
    // Legend 배경
    legend.append("rect")
      .attr("x", -10)
      .attr("y", -10)
      .attr("width", legendWidth * 2 + 20)
      .attr("height", legendHeight)
      .attr("fill", "white")
      .attr("fill-opacity", 0.95)
      .attr("stroke", "#333")
      .attr("stroke-width", 1.5)
      .attr("rx", 5);
    
    legend.append("text")
      .attr("x", legendWidth - 10)
      .attr("y", 8)
      .attr("fill", "black")
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .style("text-anchor", "middle")
      .text("Trajectory Type");
    
    // Legend items를 2열로 배치
    legendItems.forEach((item, i) => {
      const col = i < itemsPerColumn ? 0 : 1;
      const row = i < itemsPerColumn ? i : i - itemsPerColumn;
      
      const itemGroup = legend.append("g")
        .attr("transform", `translate(${col * legendWidth + 10}, ${25 + row * itemHeight})`);
      
      itemGroup.append("circle")
        .attr("cx", 5)
        .attr("cy", 0)
        .attr("r", 4)
        .attr("fill", labelColors[item.label])
        .attr("stroke", "#333")
        .attr("stroke-width", 1);
      
      itemGroup.append("text")
        .attr("x", 15)
        .attr("y", 4)
        .attr("fill", "#333")
        .style("font-size", "10px")
        .text(item.name);
    });
  }

  render() {
    return (
      <div className="latent-space-plot">
        <svg ref={this.nodeRef}></svg>
      </div>
    );
  }
}

LatentSpacePlot.propTypes = {
  data: PropTypes.arrayOf(
    PropTypes.shape({
      index: PropTypes.number.isRequired,
      x: PropTypes.number.isRequired,
      y: PropTypes.number.isRequired,
      latent: PropTypes.array.isRequired,
      trajectory: PropTypes.array.isRequired,
      label: PropTypes.string
    })
  ).isRequired,
  onHover: PropTypes.func,
  onClick: PropTypes.func,
  method: PropTypes.string,
  mode: PropTypes.string
};

export default LatentSpacePlot;
