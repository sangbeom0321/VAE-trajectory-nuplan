import React, { Component } from "react";
import PropTypes from "prop-types";
import * as d3 from "d3";
import "./TrajectoryCanvas.css";

class TrajectoryCanvas extends Component {
  constructor(props) {
    super(props);
    this.renderVisualization = this.renderVisualization.bind(this);
    this.svgRef = React.createRef();
  }

  componentDidMount() {
    this.renderVisualization();
  }

  componentDidUpdate() {
    this.renderVisualization();
  }

  renderVisualization() {
    const { trajectory, bounds } = this.props;
    if (!trajectory || !this.svgRef.current) return;

    const svg = d3.select(this.svgRef.current);
    svg.selectAll("*").remove();

    const width = 800;
    const margin = { top: 40, right: 180, bottom: 40, left: 40 };  // right margin 증가하여 legend 공간 확보

    // 고정된 범위 사용 (bounds가 제공된 경우)
    let xMin, xMax, yMin, yMax;
    
    if (bounds) {
      // 전달받은 고정 범위 사용
      xMin = bounds.xMin;
      xMax = bounds.xMax;
      yMin = bounds.yMin;
      yMax = bounds.yMax;
    } else {
      // bounds가 없으면 현재 trajectory의 범위 계산 (fallback)
      xMin = Infinity;
      xMax = -Infinity;
      yMin = Infinity;
      yMax = -Infinity;

      trajectory.forEach(point => {
        xMin = Math.min(xMin, point[0]);
        xMax = Math.max(xMax, point[0]);
        yMin = Math.min(yMin, point[1]);
        yMax = Math.max(yMax, point[1]);
      });

      // 범위가 없으면 기본 범위 설정
      if (xMin === Infinity) {
        const defaultRange = 50;
        xMin = -defaultRange;
        xMax = defaultRange;
        yMin = -defaultRange;
        yMax = defaultRange;
      }

      const padding = 10;
      xMin -= padding;
      xMax += padding;
      yMin -= padding;
      yMax += padding;
    }

    const dataWidth = xMax - xMin;
    const dataHeight = yMax - yMin;
    const aspectRatio = dataHeight / dataWidth;
    
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = Math.max(400, Math.min(800, plotWidth * aspectRatio));
    const height = plotHeight + margin.top + margin.bottom;

    svg.attr("viewBox", `0 0 ${width} ${height}`)
       .attr("preserveAspectRatio", "xMidYMid meet")
       .style("width", "100%")
       .style("height", "auto");

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear()
      .domain([xMin, xMax])
      .range([0, plotWidth]);

    const yScale = d3.scaleLinear()
      .domain([yMin, yMax])
      .range([plotHeight, 0]);

    // 경로 그리기
    const trajectoryLine = d3.line()
      .x(d => xScale(d[0]))
      .y(d => yScale(d[1]));

    // 경로 선
    g.append("path")
      .datum(trajectory)
      .attr("fill", "none")
      .attr("stroke", "#E53935")
      .attr("stroke-width", 4)
      .attr("opacity", 1.0)
      .attr("stroke-linecap", "round")
      .attr("stroke-linejoin", "round")
      .attr("d", trajectoryLine)
      .attr("class", "trajectory-path");

    // 경로 포인트 표시
    trajectory.forEach((point, i) => {
      if (i % 8 === 0) {
        g.append("circle")
          .attr("cx", xScale(point[0]))
          .attr("cy", yScale(point[1]))
          .attr("r", 3)
          .attr("fill", "#E53935")
          .attr("stroke", "white")
          .attr("stroke-width", 1);
      }
    });

    // 시작점 표시
    if (trajectory.length > 0) {
      g.append("circle")
        .attr("cx", xScale(trajectory[0][0]))
        .attr("cy", yScale(trajectory[0][1]))
        .attr("r", 5)
        .attr("fill", "green")
        .attr("stroke", "black")
        .attr("stroke-width", 1);
    }

    // 끝점 표시
    if (trajectory.length > 0) {
      g.append("circle")
        .attr("cx", xScale(trajectory[trajectory.length - 1][0]))
        .attr("cy", yScale(trajectory[trajectory.length - 1][1]))
        .attr("r", 5)
        .attr("fill", "red")
        .attr("stroke", "black")
        .attr("stroke-width", 1);
    }

    // 축 추가
    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3.axisLeft(yScale);

    g.append("g")
      .attr("transform", `translate(0,${plotHeight})`)
      .call(xAxis)
      .append("text")
      .attr("x", plotWidth / 2)
      .attr("y", 35)
      .attr("fill", "black")
      .style("text-anchor", "middle")
      .text("X (m)");

    g.append("g")
      .call(yAxis)
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -40)
      .attr("x", -plotHeight / 2)
      .attr("fill", "black")
      .style("text-anchor", "middle")
      .text("Y (m)");

    // 레전드 추가 (plotWidth 내부로 이동)
    const legend = g.append("g")
      .attr("transform", `translate(${plotWidth - 150}, 20)`);

    const legendItems = [
      { label: "Start", color: "green", stroke: false },
      { label: "End", color: "red", stroke: false },
      { label: "Trajectory", color: "#E53935", stroke: true }
    ];
    
    const legendWidth = 140;
    const legendHeight = legendItems.length * 22 + 20;

    legend.append("rect")
      .attr("x", -10)
      .attr("y", -10)
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .attr("fill", "white")
      .attr("fill-opacity", 0.95)
      .attr("stroke", "#333")
      .attr("stroke-width", 1.5)
      .attr("rx", 5)
      .style("box-shadow", "0 2px 4px rgba(0,0,0,0.2)");

    legend.append("text")
      .attr("x", 5)
      .attr("y", 8)
      .attr("fill", "black")
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .text("Legend");

    legendItems.forEach((item, i) => {
      const itemGroup = legend.append("g")
        .attr("transform", `translate(5, ${22 + i * 22})`);

      if (item.stroke) {
        itemGroup.append("line")
          .attr("x1", 5)
          .attr("x2", 25)
          .attr("y1", 0)
          .attr("y2", 0)
          .attr("stroke", item.color)
          .attr("stroke-width", 4);
      } else {
        itemGroup.append("circle")
          .attr("cx", 15)
          .attr("cy", 0)
          .attr("r", 5)
          .attr("fill", item.color)
          .attr("stroke", "black")
          .attr("stroke-width", 1);
      }

      itemGroup.append("text")
        .attr("x", 33)
        .attr("y", 4)
        .attr("fill", "#333")
        .style("font-size", "11px")
        .text(item.label);
    });
  }

  render() {
    return (
      <div className="trajectory-canvas">
        <svg ref={this.svgRef}></svg>
      </div>
    );
  }
}

TrajectoryCanvas.propTypes = {
  trajectory: PropTypes.arrayOf(
    PropTypes.arrayOf(PropTypes.number)
  ).isRequired,
  bounds: PropTypes.shape({
    xMin: PropTypes.number,
    xMax: PropTypes.number,
    yMin: PropTypes.number,
    yMax: PropTypes.number
  })
};

export default TrajectoryCanvas;
