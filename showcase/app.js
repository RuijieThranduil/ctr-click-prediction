const data = window.ctrProjectData;

const formatMetric = (value) => Number(value).toFixed(3);
let activeMetric = data.metricExplorer.defaultMetric;
let activeFeatureIndex = 0;

function renderMeta() {
  document.title = `${data.meta.title} | Project Showcase`;
  document.querySelector("[data-title]").textContent = data.meta.title;
  document.querySelector("[data-subtitle]").textContent = data.meta.subtitle;
  document.querySelector("[data-purpose]").textContent = data.purpose.question;
  document.querySelector("[data-judgment]").textContent = data.purpose.judgment;
  document.querySelector("[data-takeaway]").textContent = data.purpose.takeaway;

  const metaList = document.querySelector("[data-meta-list]");
  const items = [
    ["Audience", data.meta.audience],
    ["Updated", data.meta.updated],
    ["Source", data.meta.source],
    ["Sample", data.meta.sample],
    ["Scope", data.meta.scope]
  ];
  metaList.innerHTML = items
    .map(([label, value]) => `<li><span>${label}</span><strong>${value}</strong></li>`)
    .join("");
}

function renderHighlights() {
  document.querySelector("[data-highlights]").innerHTML = data.highlights
    .map(
      (item) => `
        <article class="metric-card">
          <span>${item.label}</span>
          <strong>${item.value}</strong>
          <p>${item.detail}</p>
        </article>
      `
    )
    .join("");
}

function renderHeroTelemetry() {
  document.querySelector("[data-hero-telemetry]").innerHTML = data.highlights
    .map(
      (item) => `
        <article class="telemetry-cell">
          <span>${item.label}</span>
          <strong>${item.value}</strong>
          <p>${item.detail}</p>
        </article>
      `
    )
    .join("");
}

function renderMetricRows() {
  document.querySelector("[data-metric-rows]").innerHTML = data.metrics
    .map(
      (row, index) => `
        <tr class="${index === data.metrics.length - 1 ? "is-best" : ""}">
          <th scope="row">${row.model}</th>
          <td>${formatMetric(row.logLoss)}</td>
          <td>${formatMetric(row.rocAuc)}</td>
          <td>${formatMetric(row.prAuc)}</td>
          <td>${row.note}</td>
        </tr>
      `
    )
    .join("");
}

function metricDefinition(key) {
  return data.metricExplorer.metrics.find((metric) => metric.key === key);
}

function renderMetricSwitcher() {
  const switcher = document.querySelector("[data-metric-switcher]");
  switcher.innerHTML = data.metricExplorer.metrics
    .map(
      (metric) => `
        <button type="button" class="mini-button ${metric.key === activeMetric ? "is-active" : ""}" data-metric="${metric.key}">
          ${metric.label}
        </button>
      `
    )
    .join("");
  switcher.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-metric]");
    if (!button) return;
    activeMetric = button.dataset.metric;
    renderMetricSwitcher();
    renderMetricChart();
  }, { once: true });
}

function renderMetricChart() {
  const metric = metricDefinition(activeMetric);
  const [min, max] = metric.domain;
  const range = max - min;
  document.querySelector("[data-metric-chart]").innerHTML = data.metrics
    .map((row, index) => {
      const value = row[activeMetric];
      const width = Math.max(4, Math.min(100, ((value - min) / range) * 100));
      const isBest = activeMetric === "logLoss"
        ? value === Math.min(...data.metrics.map((item) => item[activeMetric]))
        : value === Math.max(...data.metrics.map((item) => item[activeMetric]));
      return `
        <button type="button" class="metric-bar ${isBest ? "is-best" : ""}" data-model-index="${index}">
          <span class="bar-label">${row.model}</span>
          <span class="bar-track"><span style="width: ${width}%"></span></span>
          <strong>${formatMetric(value)}</strong>
        </button>
      `;
    })
    .join("");
  document.querySelector("[data-metric-summary]").textContent = `${metric.label}: ${metric.summary} (${metric.direction}).`;
}

function renderFeatureChart(selectedIndex = activeFeatureIndex) {
  activeFeatureIndex = selectedIndex;
  const max = Math.max(...data.featureImportance.map((item) => item.importance));
  const selected = data.featureImportance[selectedIndex];
  document.querySelector("[data-feature-chart]").innerHTML = data.featureImportance
    .map((item, index) => {
      const width = Math.max(4, (item.importance / max) * 100);
      return `
        <button type="button" class="feature-bar ${index === selectedIndex ? "is-active" : ""}" data-feature-index="${index}">
          <span>${item.feature}</span>
          <span class="bar-track"><span style="width: ${width}%"></span></span>
          <strong>${item.importance.toFixed(3)}</strong>
        </button>
      `;
    })
    .join("");
  document.querySelector("[data-feature-note]").textContent = `${selected.feature} - ${selected.group}. ${selected.note}`;
}

function bindFeatureChart() {
  const chart = document.querySelector("[data-feature-chart]");
  chart.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-feature-index]");
    if (!button) return;
    renderFeatureChart(Number(button.dataset.featureIndex));
  });
}

function bindScrollEffects() {
  const progress = document.querySelector(".scroll-progress");
  const updateProgress = () => {
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const percent = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;
    progress.style.width = `${Math.min(100, Math.max(0, percent))}%`;
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
      }
    });
  }, { threshold: 0.16 });

  document.querySelectorAll(".reveal").forEach((element) => observer.observe(element));
  window.addEventListener("scroll", updateProgress, { passive: true });
  updateProgress();
}

function initSignalField() {
  const canvas = document.querySelector("[data-signal-field]");
  if (!canvas) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const context = canvas.getContext("2d");
  const points = Array.from({ length: 46 }, (_, index) => ({
    x: (index * 83) % 100,
    y: (index * 47) % 100,
    speed: 0.08 + (index % 5) * 0.014,
    radius: 1 + (index % 3) * 0.45
  }));

  const resize = () => {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(canvas.clientWidth * ratio);
    canvas.height = Math.floor(canvas.clientHeight * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
  };

  const draw = (time = 0) => {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    context.clearRect(0, 0, width, height);
    context.lineWidth = 1;

    const plotted = points.map((point) => {
      const drift = reducedMotion ? 0 : Math.sin(time * 0.00018 + point.x) * 18;
      return {
        x: (point.x / 100) * width + drift,
        y: ((point.y + (reducedMotion ? 0 : time * point.speed * 0.01)) % 100 / 100) * height,
        radius: point.radius
      };
    });

    plotted.forEach((point, index) => {
      context.beginPath();
      context.fillStyle = index % 4 === 0 ? "rgba(193, 140, 68, 0.72)" : "rgba(119, 215, 226, 0.62)";
      context.arc(point.x, point.y, point.radius, 0, Math.PI * 2);
      context.fill();
    });

    for (let i = 0; i < plotted.length; i += 1) {
      for (let j = i + 1; j < plotted.length; j += 1) {
        const dx = plotted[i].x - plotted[j].x;
        const dy = plotted[i].y - plotted[j].y;
        const distance = Math.hypot(dx, dy);
        if (distance < 150) {
          context.beginPath();
          context.strokeStyle = `rgba(119, 215, 226, ${0.18 * (1 - distance / 150)})`;
          context.moveTo(plotted[i].x, plotted[i].y);
          context.lineTo(plotted[j].x, plotted[j].y);
          context.stroke();
        }
      }
    }

    if (!reducedMotion) {
      requestAnimationFrame(draw);
    }
  };

  resize();
  window.addEventListener("resize", resize);
  draw();
}

function renderWorkflow() {
  document.querySelector("[data-workflow]").innerHTML = data.workflow
    .map(
      (item) => `
        <li class="workflow-step">
          <span class="step-index">${item.step}</span>
          <div>
            <h3>${item.title}</h3>
            <p>${item.body}</p>
            <small>${item.artifact}</small>
          </div>
        </li>
      `
    )
    .join("");
}

function renderChartFilters() {
  const filterWrap = document.querySelector("[data-chart-filters]");
  const types = ["All", ...new Set(data.charts.map((chart) => chart.type))];
  filterWrap.innerHTML = types
    .map(
      (type, index) => `
        <button type="button" class="filter-button ${index === 0 ? "is-active" : ""}" data-filter="${type}">
          ${type}
        </button>
      `
    )
    .join("");
  filterWrap.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    filterWrap.querySelectorAll("button").forEach((item) => item.classList.remove("is-active"));
    button.classList.add("is-active");
    renderCharts(button.dataset.filter);
  });
}

function renderCharts(filter = "All") {
  const charts = filter === "All" ? data.charts : data.charts.filter((chart) => chart.type === filter);
  document.querySelector("[data-charts]").innerHTML = charts
    .map(
      (chart) => `
        <article class="chart-card">
          <div class="chart-copy">
            <span>${chart.type}</span>
            <h3>${chart.title}</h3>
            <p>${chart.insight}</p>
          </div>
          <img src="${chart.src}" alt="${chart.alt}" loading="lazy">
        </article>
      `
    )
    .join("");
}

function renderNotes() {
  document.querySelector("[data-notes]").innerHTML = data.methodNotes
    .map((note) => `<li>${note}</li>`)
    .join("");
  document.querySelector("[data-contract]").innerHTML = `
    <p><strong>${data.dataContract.goal}</strong></p>
    <p>${data.dataContract.payload}</p>
    <p>${data.dataContract.reuse}</p>
  `;
  document.querySelector("[data-repository]").href = data.meta.repository;
}

renderMeta();
renderHighlights();
renderHeroTelemetry();
renderMetricRows();
renderMetricSwitcher();
renderMetricChart();
renderFeatureChart();
bindFeatureChart();
renderWorkflow();
renderChartFilters();
renderCharts();
renderNotes();
bindScrollEffects();
initSignalField();
