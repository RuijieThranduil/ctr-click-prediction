const data = window.ctrProjectData;

const formatMetric = (value) => Number(value).toFixed(3);

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
renderMetricRows();
renderWorkflow();
renderChartFilters();
renderCharts();
renderNotes();
