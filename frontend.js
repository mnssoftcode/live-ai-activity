const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const resultsDiv = document.getElementById('results');
const summaryDiv = document.getElementById('summary');
const resultsTable = document.getElementById('results-table');
const resultsTbody = resultsTable.querySelector('tbody');
const exportBtn = document.getElementById('export-btn');
const timelineChartCanvas = document.getElementById('timeline-chart');

let streaming = false;
let intervalId = null;
let detectionHistory = [];
let chart = null;

// For smooth UI updates
let lastStableData = null;
let lastStableTime = 0;
let lastTableHTML = '';
let lastSummaryHTML = '';
let lastDetectionData = null;
let lastDetectionTime = 0;
const TABLE_UPDATE_INTERVAL = 1000; // ms
const DETECTION_INTERVAL = 500; // ms

async function startVideo() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();
}

function getCurrentTimeString() {
  const now = new Date();
  return now.toLocaleTimeString();
}

function getCroppedThumb([x1, y1, x2, y2]) {
  const tempCanvas = document.createElement('canvas');
  const w = Math.max(1, x2 - x1), h = Math.max(1, y2 - y1);
  tempCanvas.width = w;
  tempCanvas.height = h;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(video, x1, y1, w, h, 0, 0, w, h);
  return tempCanvas.toDataURL('image/jpeg');
}

async function sendFrame() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
  const formData = new FormData();
  formData.append('file', blob, 'frame.jpg');

  try {
    const res = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    const now = getCurrentTimeString();
    // Add detection to history for timeline and export
    if (data.count > 0) {
      for (let i = 0; i < data.count; i++) {
        detectionHistory.push({
          time: now,
          activity: data.labels[i],
          confidence: data.scores[i],
          box: data.boxes[i],
          thumb: getCroppedThumb(data.boxes[i])
        });
      }
    }
    // For overlay, always update
    drawResults(data);
    // For table/summary, only update at most every 1s, and only if changed
    lastDetectionData = { ...data, now };
    lastDetectionTime = Date.now();
  } catch (err) {
    console.error('Prediction error:', err);
  }
}

function drawResults(data) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!data.boxes) return;
  for (let i = 0; i < data.boxes.length; i++) {
    const [x1, y1, x2, y2] = data.boxes[i];
    const label = data.labels[i] || 'person';
    const score = data.scores[i] || 0;
    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = 'rgba(0,255,0,0.7)';
    ctx.font = '16px Arial';
    ctx.fillText(`${label} (${(score*100).toFixed(1)}%)`, x1, y1 - 5);
  }
}

function stableUpdateUI() {
  // Use the most recent detection, but only update table/summary every 1s
  const now = Date.now();
  if (!lastDetectionData) return;
  // If new detection data is available and different, update
  if (
    !lastStableData ||
    JSON.stringify(lastDetectionData) !== JSON.stringify(lastStableData) ||
    now - lastStableTime > TABLE_UPDATE_INTERVAL
  ) {
    lastStableData = lastDetectionData;
    lastStableTime = now;
    showResults(lastStableData, lastStableData.now);
    updateTimeline();
  }
}

function showResults(data, now) {
  // Summary
  let summaryHTML = `<b>People detected:</b> ${data.count || 0}`;
  if (data.labels && data.labels.length) {
    // Activity summary
    const activityCounts = {};
    data.labels.forEach(l => { activityCounts[l] = (activityCounts[l] || 0) + 1; });
    summaryHTML += '<br><b>Activities:</b> ' + Object.entries(activityCounts).map(([act, cnt]) => `${act} (${cnt})`).join(', ');
  }
  // Only update if changed
  if (summaryHTML !== lastSummaryHTML) {
    summaryDiv.innerHTML = summaryHTML;
    lastSummaryHTML = summaryHTML;
  }

  // Table
  let tableHTML = '';
  if (data.count > 0 && data.boxes && data.labels && data.scores) {
    resultsTable.style.display = '';
    for (let i = 0; i < data.count; i++) {
      const thumbDataUrl = getCroppedThumb(data.boxes[i]);
      const conf = data.scores[i] || 0;
      const confBar = `<div class="conf-bar"><div class="conf-bar-inner" style="width:${(conf*100).toFixed(1)}%"></div></div>`;
      tableHTML += `
        <tr style="opacity:0; transition:opacity 0.4s;">
          <td>${i + 1}</td>
          <td><img class="thumb" src="${thumbDataUrl}" alt="thumb"></td>
          <td>${data.labels[i]}</td>
          <td>${confBar} <span style="font-size:0.95em;">${(conf * 100).toFixed(1)}%</span></td>
          <td>[${data.boxes[i].map(x => Math.round(x)).join(', ')}]</td>
          <td>${now}</td>
        </tr>
      `;
    }
    // Only update if changed
    if (tableHTML !== lastTableHTML) {
      resultsTbody.innerHTML = tableHTML;
      // Fade in effect
      setTimeout(() => {
        Array.from(resultsTbody.children).forEach(row => row.style.opacity = 1);
      }, 10);
      lastTableHTML = tableHTML;
    }
  } else {
    resultsTable.style.display = 'none';
    resultsTbody.innerHTML = '';
    lastTableHTML = '';
  }
}

exportBtn.onclick = function() {
  if (!detectionHistory.length) return;
  let csv = 'Time,Activity,Confidence,Box,Thumbnail\n';
  detectionHistory.forEach(d => {
    csv += `${d.time},${d.activity},${(d.confidence*100).toFixed(1)}%,[${d.box.map(x=>Math.round(x)).join(' ')}],${d.thumb}\n`;
  });
  const blob = new Blob([csv], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'detection_results.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

function updateTimeline() {
  const timeBuckets = {};
  detectionHistory.forEach(d => {
    if (!timeBuckets[d.time]) timeBuckets[d.time] = {};
    timeBuckets[d.time][d.activity] = (timeBuckets[d.time][d.activity] || 0) + 1;
  });
  const times = Object.keys(timeBuckets).slice(-20);
  const activities = Array.from(new Set(detectionHistory.map(d => d.activity)));
  const datasets = activities.map((act, idx) => ({
    label: act,
    data: times.map(t => timeBuckets[t][act] || 0),
    backgroundColor: `hsl(${(idx*60)%360},70%,60%)`,
    borderColor: `hsl(${(idx*60)%360},70%,40%)`,
    fill: true,
    tension: 0.3
  }));
  if (!chart) {
    chart = new Chart(timelineChartCanvas, {
      type: 'line',
      data: { labels: times, datasets },
      options: {
        responsive: true,
        plugins: { legend: { display: true } },
        scales: { y: { beginAtZero: true, precision: 0 } }
      }
    });
  } else {
    chart.data.labels = times;
    chart.data.datasets = datasets;
    chart.update();
  }
}

startBtn.onclick = async () => {
  if (streaming) return;
  streaming = true;
  await startVideo();
  intervalId = setInterval(sendFrame, DETECTION_INTERVAL);
  setInterval(stableUpdateUI, TABLE_UPDATE_INTERVAL);
  startBtn.disabled = true;
  startBtn.textContent = 'Running...';
}; 