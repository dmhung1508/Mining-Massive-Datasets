// AI News Anchor: Live2D presenter + TV-style broadcast UI driven by /broadcast and /tts.

import { runEarthIntro } from "./intro.js";

const API_BASE = (location.origin && location.origin.startsWith("http"))
  ? location.origin
  : "http://127.0.0.1:8765";

const MODEL_URL = `${API_BASE}/avatar/ptit_sdk.model3.json`;
const BG_URL = `${API_BASE}/avatar/background/bg.png`;

const state = {
  app: null,
  model: null,
  segments: [],
  news: [],
  index: 0,
  playing: false,
  audio: null,
};

const els = {
  status: document.getElementById("status"),
  bg: document.getElementById("bg"),
  intro: document.getElementById("intro"),
  introTitle: document.getElementById("introTitle"),
  pip: document.getElementById("pip"),
  pipImg: document.getElementById("pipImg"),
  ltTopic: document.getElementById("ltTopic"),
  ltHeadline: document.getElementById("ltHeadline"),
  ltSub: document.getElementById("ltSub"),
  tickerText: document.getElementById("tickerText"),
  clock: document.getElementById("clock"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  topN: document.getElementById("topN"),
  useLlm: document.getElementById("useLlm"),
};

const TOPIC_LABEL = {
  russia_ukraine_war: "XUNG ĐỘT NGA - UKRAINE",
  us_iran_war: "CĂNG THẲNG MỸ - IRAN",
  unknown: "TIN TỔNG HỢP",
};

function setStatus(t) { els.status.textContent = t; }

function startClock() {
  const tick = () => {
    const d = new Date();
    els.clock.textContent = d.toLocaleTimeString("vi-VN", { hour12: false });
  };
  tick();
  setInterval(tick, 1000);
}

els.bg.style.backgroundImage = `url(${BG_URL})`;

async function initLive2D() {
  if (state.app) return;
  const canvas = document.getElementById("live2d");
  state.app = new PIXI.Application({
    view: canvas,
    autoStart: true,
    resizeTo: canvas.parentElement,
    backgroundAlpha: 0,
  });
  setStatus("Đang tải người dẫn Ami...");
  try {
    state.model = await PIXI.live2d.Live2DModel.from(MODEL_URL);
    state.app.stage.addChild(state.model);
    fitModel();
    window.addEventListener("resize", fitModel);
    setStatus("Sẵn sàng. Nhấn Bắt đầu.");
  } catch (err) {
    console.error(err);
    setStatus("Không tải được người dẫn Live2D.");
  }
}

function fitModel() {
  if (!state.model || !state.app) return;
  const { width, height } = state.app.renderer.screen;
  const scale = Math.min(width / state.model.width, height / state.model.height) * 1.15;
  state.model.scale.set(scale);
  state.model.anchor.set(0.5, 0.5);
  state.model.x = width / 2;
  state.model.y = height / 2 + height * 0.12; // sit slightly low so head is framed
}

function topicLabel(topic) {
  return TOPIC_LABEL[(topic || "").toLowerCase()] || "TIN TỔNG HỢP";
}

function newsForCluster(clusterId) {
  return state.news.find((n) => n.cluster_id === clusterId);
}

function updateLowerThird(seg) {
  if (seg.kind === "story") {
    const n = newsForCluster(seg.cluster_id) || {};
    els.ltTopic.textContent = topicLabel(n.topic);
    // Show Vietnamese keywords/entities as the headline instead of raw English tweets.
    const entities = (n.entities || []).slice(0, 4).join(" · ");
    els.ltHeadline.textContent = entities || `Tin số ${seg.cluster_id}`;
    els.ltSub.textContent = `Cụm gồm ${n.cluster_size || "?"} bài đăng gần trùng lặp`;
  } else if (seg.kind === "intro") {
    els.ltTopic.textContent = "BẢN TIN TỔNG HỢP";
    els.ltHeadline.textContent = "Điểm tin nổi bật từ mạng xã hội";
    els.ltSub.textContent = "Phát hiện cụm tin bằng MinHash + LSH";
  } else {
    els.ltTopic.textContent = "KẾT THÚC";
    els.ltHeadline.textContent = "Cảm ơn quý vị đã theo dõi";
    els.ltSub.textContent = "";
  }
}

function showImage(seg) {
  if (seg.image_path) {
    els.pipImg.style.backgroundImage = `url(${API_BASE}${seg.image_path})`;
    els.pip.classList.add("show");
  } else {
    els.pip.classList.remove("show");
  }
}

function buildTicker() {
  // Use the varied Vietnamese story sentences so the ticker isn't repetitive.
  const lines = state.segments
    .filter((s) => s.kind === "story")
    .map((s, i) => `TIN ${i + 1}: ${s.text}`)
    .filter(Boolean);
  if (lines.length) {
    els.tickerText.textContent = lines.join("       ◆       ") + "       ◆       ";
  }
}

async function fetchBroadcast() {
  const topN = parseInt(els.topN.value, 10) || 5;
  const useLlm = els.useLlm.checked;
  setStatus("Đang lấy tin đã phân cụm...");
  const res = await fetch(`${API_BASE}/broadcast?top_n=${topN}&use_llm=${useLlm}`);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `broadcast failed (${res.status})`);
  }
  return res.json();
}

async function synthesize(text) {
  const res = await fetch(`${API_BASE}/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`tts failed (${res.status})`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

function speakWithLipSync(audioUrl) {
  return new Promise((resolve, reject) => {
    const audio = new Audio(audioUrl);
    state.audio = audio;

    let ctx, analyser, dataArray, source, raf;
    try {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      source = ctx.createMediaElementSource(audio);
      analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyser.connect(ctx.destination);
      dataArray = new Uint8Array(analyser.frequencyBinCount);
    } catch (e) {
      console.warn("WebAudio unavailable, no lip-sync", e);
    }

    const tick = () => {
      if (analyser && state.model) {
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (const v of dataArray) sum += v;
        const level = Math.min(1, (sum / dataArray.length) / 80);
        try {
          state.model.internalModel.coreModel.setParameterValueById("ParamMouthOpenY", level);
        } catch (_) {}
      }
      raf = requestAnimationFrame(tick);
    };

    audio.onplay = () => { if (ctx && ctx.state === "suspended") ctx.resume(); tick(); };
    audio.onended = () => {
      cancelAnimationFrame(raf);
      if (state.model) {
        try { state.model.internalModel.coreModel.setParameterValueById("ParamMouthOpenY", 0); } catch (_) {}
      }
      URL.revokeObjectURL(audioUrl);
      resolve();
    };
    audio.onerror = (e) => { cancelAnimationFrame(raf); reject(e); };
    audio.play().catch(reject);
  });
}

async function playLoop() {
  for (; state.index < state.segments.length; state.index++) {
    if (!state.playing) break;
    const seg = state.segments[state.index];
    updateLowerThird(seg);
    showImage(seg);
    setStatus(`Đang đọc (${state.index + 1}/${state.segments.length})`);
    try {
      const url = await synthesize(seg.text);
      await speakWithLipSync(url);
    } catch (err) {
      console.error(err);
      setStatus(`Lỗi đoạn ${state.index + 1}: ${err.message}`);
    }
  }
  if (state.playing) setStatus("Bản tin kết thúc.");
  stop();
}

async function start() {
  els.startBtn.classList.add("hidden");
  try {
    // Fetch the broadcast in the background; capture errors so a fetch failure
    // never blocks the intro animation.
    const dataPromise = fetchBroadcast().catch((err) => ({ __error: err }));

    // Phase 1: spin the Earth for ~3 seconds.
    setStatus("Quả địa cầu đang quay...");
    try {
      await runEarthIntro({ container: els.intro, spins: 2, duration: 5000 });
    } catch (introErr) {
      console.error("Intro error:", introErr);
      setStatus("Bỏ qua intro (WebGL lỗi): " + introErr.message);
    }

    // Phase 2: show the title card for ~2 seconds.
    els.introTitle.classList.add("show");
    setStatus("Chuẩn bị bản tin...");
    await sleep(2000);

    // Phase 3: hide the intro overlay and start the broadcast.
    els.intro.classList.add("hidden");
    await initLive2D();
    const data = await dataPromise;
    if (data && data.__error) throw data.__error;
    state.segments = (data && data.segments) || [];
    state.news = (data && data.news) || [];
    if (!state.segments.length) { setStatus("Không có tin nào."); return; }
    buildTicker();
    state.index = 0;
    state.playing = true;
    els.stopBtn.disabled = false;
    playLoop();
  } catch (err) {
    console.error(err);
    setStatus(`Lỗi: ${err.message}`);
    els.startBtn.classList.remove("hidden");
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function stop() {
  state.playing = false;
  if (state.audio) { state.audio.pause(); state.audio = null; }
  els.pip.classList.remove("show");
  els.stopBtn.disabled = true;
  // Bring back the intro + start button so the user can replay.
  els.introTitle.classList.remove("show");
  els.intro.classList.remove("hidden");
  els.startBtn.classList.remove("hidden");
}

els.startBtn.addEventListener("click", start);
els.stopBtn.addEventListener("click", () => { stop(); setStatus("Đã dừng."); });

startClock();
setStatus("Sẵn sàng. Nhấn Bắt đầu để mở bản tin.");
