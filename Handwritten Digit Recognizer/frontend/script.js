const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
canvas.width = canvas.offsetWidth;
canvas.height = canvas.offsetHeight;

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => {
  drawing = false;
  ctx.beginPath();
});
canvas.addEventListener("mousemove", draw);

function draw(e) {
  if (!drawing) return;
  ctx.lineWidth = 20;
  ctx.lineCap = "round";

  // Stroke color based on theme
  ctx.strokeStyle = document.body.classList.contains("dark") ? "white" : "black";

  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
}

function clearCanvas() {
  ctx.fillStyle = document.body.classList.contains("dark") ? "#111" : "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  document.getElementById("result").innerText = "";
}

async function predictDigit() {
  const image = canvas.toDataURL("image/png");
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image })
  });
  const data = await response.json();
  document.getElementById("result").innerText = "Prediction: " + data.prediction;
}

const toggleBtn = document.getElementById("theme-toggle");
toggleBtn.addEventListener("click", () => {
  document.body.classList.toggle("dark");
  clearCanvas();
});

// Info panel toggle
const infoToggle = document.getElementById("info-toggle");
const infoPanel = document.getElementById("info-panel");
const closeInfo = document.getElementById("close-info");

infoToggle.addEventListener("click", () => {
  infoPanel.classList.toggle("open");
});

closeInfo.addEventListener("click", () => {
  infoPanel.classList.remove("open");
});