// CheckGraph landing — scroll reveal + decorative graph background.
// No frameworks: single static page, kept dependency-free on purpose.

document.addEventListener("DOMContentLoaded", () => {
  const revealEls = document.querySelectorAll(".reveal");
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );
  revealEls.forEach((el) => io.observe(el));

  buildGraphBackground();
  wireContactForm();
  initTimelineProgress();
});

function initTimelineProgress() {
  const timelines = document.querySelectorAll(".timeline");
  if (!timelines.length) return;

  timelines.forEach((tl) => {
    const bar = document.createElement("div");
    bar.className = "timeline-progress";
    tl.appendChild(bar);
  });

  const update = () => {
    const viewportH = window.innerHeight;
    timelines.forEach((tl) => {
      const bar = tl.querySelector(".timeline-progress");
      const rect = tl.getBoundingClientRect();
      const start = viewportH * 0.9;
      const end = viewportH * 0.35;
      const span = rect.height + (start - end);
      const scrolled = start - rect.top;
      const pct = Math.max(0, Math.min(1, scrolled / span));
      bar.style.height = pct * 100 + "%";
    });
  };

  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
}

function buildGraphBackground() {
  const container = document.getElementById("graph-bg");
  if (!container) return;

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
  container.appendChild(svg);

  const width = window.innerWidth;
  const height = Math.max(window.innerHeight, 800);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const count = width < 720 ? 18 : 34;
  const points = Array.from({ length: count }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
  }));

  const maxDist = width < 720 ? 160 : 220;
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dx = points[i].x - points[j].x;
      const dy = points[i].y - points[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < maxDist) {
        const line = document.createElementNS(ns, "line");
        line.setAttribute("x1", points[i].x);
        line.setAttribute("y1", points[i].y);
        line.setAttribute("x2", points[j].x);
        line.setAttribute("y2", points[j].y);
        svg.appendChild(line);
      }
    }
  }

  points.forEach((p, i) => {
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("class", "node");
    circle.setAttribute("cx", p.x);
    circle.setAttribute("cy", p.y);
    circle.setAttribute("r", 2.5);
    circle.style.animationDelay = `${(i % 7) * 0.9}s`;
    circle.style.transformOrigin = `${p.x}px ${p.y}px`;
    svg.appendChild(circle);
  });
}

function wireContactForm() {
  const form = document.querySelector("form.contact");
  if (!form) return;

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const data = new FormData(form);

    // FormData posted as-is (browser sets the multipart boundary itself) so
    // file fields like the CSV upload survive the submission — a manually
    // built urlencoded body silently drops file inputs.
    fetch("/", {
      method: "POST",
      body: data,
    })
      .then(() => {
        form.hidden = true;
        document.querySelector(".form-success").style.display = "block";
      })
      .catch(() => {
        form.querySelector(".form-note").textContent =
          "No se pudo enviar. Escríbenos directamente a hola@checkgraph.dev.";
      });
  });
}
