/* ==========================================================================
   Detection workspace
   ========================================================================== */
(function () {
  "use strict";

  const {
    apiFetch,
    toast,
    escapeHtml,
    formatTimestamp,
    formatConfidence,
    confidencePercent,
  } = window.FND;

  const form = document.getElementById("detectForm");
  if (!form) return;

  const textarea = document.getElementById("newsText");
  const charCount = document.getElementById("charCount");
  const clearBtn = document.getElementById("clearBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const btnLabel = analyzeBtn.querySelector(".btn-label");
  const btnLoading = analyzeBtn.querySelector(".btn-loading");
  const resultEmpty = document.getElementById("resultEmpty");
  const resultContent = document.getElementById("resultContent");
  const verdictBanner = document.getElementById("verdictBanner");
  const verdictKicker = document.getElementById("verdictKicker");
  const verdictLabel = document.getElementById("verdictLabel");
  const verdictSub = document.getElementById("verdictSub");
  const gauge = document.getElementById("confidenceGauge");
  const gaugeCircle = document.getElementById("gaugeCircle");
  const gaugeValue = document.getElementById("gaugeValue");
  const explanationText = document.getElementById("explanationText");
  const breakdown = document.getElementById("modelBreakdown");
  const metaModel = document.getElementById("metaModel");
  const metaTokens = document.getElementById("metaTokens");
  const metaCoverage = document.getElementById("metaCoverage");
  const metaLatency = document.getElementById("metaLatency");
  const reliabilityNote = document.getElementById("reliabilityNote");
  const reliabilityText = document.getElementById("reliabilityText");
  const timestampChip = document.getElementById("resultTimestamp");
  const disclaimerText = document.getElementById("disclaimerText");
  const anotherBtn = document.getElementById("anotherBtn");
  const recentList = document.getElementById("recentList");
  const hint = document.getElementById("inputHelp");
  const MAX_CHARS = Number(textarea.getAttribute("maxlength")) || 20000;
  const CIRCUMFERENCE = 2 * Math.PI * 52;

  /* ----------------------------- Samples ----------------------------- */
  // Genuine articles from the dataset's held-out test split (title + body).
  const SAMPLES = {
    policy: {
      label: "Policy",
      text:
        "Netherlands to hold referendum on new surveillance law. AMSTERDAM (Reuters) - The Netherlands will hold a referendum on legislation giving law enforcement authorities far-reaching surveillance powers, the Dutch Voting Commission said on Wednesday. A range of activists, politicians and media groups want to overturn the tapping law , which was passed by parliament in July and allows agencies to gather data covertly from large groups of people at once. Dutch law obliges the government to hold a non-binding referendum if a petition with at least 300,000 signatures is submitted to the Voting Commission. The agency said it had received more than 384,000 valid signatures and that the referendum would be held on March 21 together with municipal elections. Opponents argue that innocent civilians will be dragged into the sights of intelligence agencies, while the government says the powers are needed to counter national security threats. The referendum",
    },
    consumer: {
      label: "Consumer claim",
      text:
        "SAY WHAT? AMAZON TELLS CUSTOMER THEY WERE FORCED BY FEDERAL GOVERNMENT To Remove Confederate Flag From Website. When it comes to limiting our free speech, nothing this lawless government does should surprise us. Under normal circumstances, I would question the validity of this claim, but we are no longer living under normal circumstances Amazon.com staff are telling irate customers that the company was ordered by the federal government not to sell items featuring the Confederate flag in the aftermath of the Charleston shooting.Amazon made the announcement this week, along with eBay, Sears and Walmart, that it would no longer sell products bearing the rebel flag, but according to a conversation posted on YouTube between a customer and an Amazon sales rep, the decision could have been made as a result of pressure from the government.At first the Amazon staffer claims that the items were banned because they were deemed to be offensive, but when pressed by the customer, the sales rep tells a different",
    },
    world: {
      label: "World",
      text:
        "Mozambique president replaces energy and foreign ministers. MAPUTO (Reuters) - Mozambican President Filipe Nyusi has sacked four ministers, including those with the foreign affairs and energy portfolios, his office said late on Tuesday, without giving a reason. Energy is a key portfolio in Mozambique, which has vast untapped offshore gas reserves that are being developed by oil majors such as Italy s Eni. Nyusi s office said Ernesto Max Elias Tonela had replaced Leticia da Silva Klemens as minister of energy and mineral resources and Jose Condugua Antonio Pacheco was the new foreign minister, replacing Oldemiro Baloi. The president also replaced the ministers of industry and trade and of agriculture and food security. Tonela, the new energy minister, previously served as commerce minister. An economist by training, Tonela has also worked previously on the board of the Hidroelectrica de Cahora Bassa (HCB) company responsible for Mozambique s 2,000",
    },
  };

  /* ----------------------------- Textarea UI ----------------------------- */
  function updateCharCount() {
    const length = textarea.value.length;
    charCount.textContent = `${length.toLocaleString()} / ${MAX_CHARS.toLocaleString()} characters`;
    charCount.classList.toggle("is-warning", length > MAX_CHARS * 0.85 && length <= MAX_CHARS);
    charCount.classList.toggle("is-over", length > MAX_CHARS);
  }

  textarea.addEventListener("input", updateCharCount);
  updateCharCount();

  clearBtn.addEventListener("click", () => {
    textarea.value = "";
    updateCharCount();
    textarea.focus();
    hideError();
  });

  document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => {
      const sample = SAMPLES[button.dataset.sample];
      if (!sample) return;
      textarea.value = sample.text;
      updateCharCount();
      textarea.focus();
      document.querySelectorAll("[data-sample]").forEach((b) => b.classList.remove("is-active"));
      button.classList.add("is-active");
      hideError();
    });
  });

  /* ----------------------------- Model choice ----------------------------- */
  const params = new URLSearchParams(window.location.search);
  const requestedModel = params.get("model");
  if (requestedModel) {
    const input = form.querySelector(`input[name="model"][value="${CSS.escape(requestedModel)}"]`);
    if (input) input.checked = true;
  }

  function selectedModel() {
    const checked = form.querySelector('input[name="model"]:checked');
    return checked ? checked.value : "bilstm";
  }

  /* ----------------------------- Errors ----------------------------- */
  function showError(message) {
    hint.classList.add("is-error");
    hint.textContent = message;
    toast(message, "error");
  }
  function hideError() {
    hint.classList.remove("is-error");
    hint.innerHTML =
      "Minimum 15 characters · maximum 20,000 characters · longer articles are truncated to the first 256 tokens.";
  }

  /* ----------------------------- Loading state ----------------------------- */
  function setLoading(loading) {
    analyzeBtn.disabled = loading;
    btnLabel.hidden = loading;
    btnLoading.hidden = !loading;
    textarea.readOnly = loading;
    if (loading) {
      resultEmpty.hidden = false;
      resultContent.hidden = true;
      resultEmpty.querySelector("h3").textContent = "Analysing your article…";
      resultEmpty.querySelector("p").textContent =
        "Cleaning the text, tokenising it and running the neural network.";
      resultEmpty.querySelector(".radar-ring").style.animationDuration = "1.2s";
    } else {
      resultEmpty.querySelector("h3").textContent = "Awaiting analysis";
      resultEmpty.querySelector("p").textContent =
        "Your classification, confidence score and a per-model breakdown will appear here.";
    }
  }

  /* ----------------------------- Validation ----------------------------- */
  function validate(text) {
    const trimmed = text.trim();
    if (!trimmed) {
      showError("Please enter some news text before analysing.");
      return false;
    }
    if (trimmed.length < 15) {
      showError("That input is too short. Please paste at least a headline or a full sentence.");
      return false;
    }
    if (trimmed.length > MAX_CHARS) {
      showError(`Input exceeds the ${MAX_CHARS.toLocaleString()} character limit.`);
      return false;
    }
    return true;
  }

  /* ----------------------------- Rendering ----------------------------- */
  function animateGauge(confidence, label) {
    const target = confidencePercent(confidence);
    const offset = CIRCUMFERENCE * (1 - target / 100);
    gauge.classList.remove("is-fake", "is-real");
    gauge.classList.add(label === "FAKE" ? "is-fake" : "is-real");
    gaugeCircle.style.strokeDasharray = String(CIRCUMFERENCE);
    gaugeCircle.style.strokeDashoffset = String(CIRCUMFERENCE);
    // Force reflow so the transition always replays.
    void gaugeCircle.getBoundingClientRect();
    requestAnimationFrame(() => {
      gaugeCircle.style.strokeDashoffset = String(offset);
    });
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / 900, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      gaugeValue.textContent = `${(target * eased).toFixed(1)}%`;
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function renderResult(data) {
    const isFake = data.prediction === "FAKE";
    verdictBanner.classList.remove("is-fake", "is-real");
    verdictBanner.classList.add(isFake ? "is-fake" : "is-real");
    verdictKicker.textContent = isFake ? "Pattern detected" : "Pattern detected";
    verdictLabel.textContent = `${data.prediction} NEWS`;
    const safety = data.reliability && data.reliability.level !== "high"
      ? " Low-context input — treat this confidence with caution."
      : "";
    verdictSub.textContent = isFake
      ? "Language patterns align with fabricated articles in the training corpus." + safety
      : "Language patterns align with professional reporting in the training corpus." + safety;

    gaugeValue.textContent = "0%";
    animateGauge(Number(data.confidence) || 0, data.prediction);

    explanationText.textContent = data.explanation || "";
    timestampChip.textContent = formatTimestamp(data.timestamp) || "Just now";
    metaModel.textContent = data.model_used || "—";
    const stats = data.text_stats || {};
    metaTokens.textContent = `${stats.tokens_used ?? "—"} / ${stats.max_len ?? 256}${
      stats.truncated ? " (truncated)" : ""
    }`;
    metaCoverage.textContent =
      stats.vocabulary_coverage != null
        ? `${(stats.vocabulary_coverage * 100).toFixed(1)}%`
        : "—";
    metaLatency.textContent = `${Number(data.processing_ms || 0).toFixed(0)} ms`;

    const reliability = data.reliability || { level: "medium", message: "" };
    reliabilityNote.classList.remove("is-low", "is-high", "is-medium");
    reliabilityNote.classList.add(`is-${reliability.level}`);
    reliabilityText.textContent = reliability.message;

    const selectedKey = data.model_key;
    breakdown.innerHTML = (data.model_scores || [])
      .map((score) => {
        const fake = score.prediction === "FAKE";
        const isSelected = score.key === selectedKey;
        return `
          <div class="breakdown-row">
            <div class="breakdown-head">
              <span class="breakdown-name">
                <span class="dot ${fake ? "dot-red" : "dot-green"}"></span>
                ${escapeHtml(score.label)}
                ${isSelected ? '<span class="badge badge-best">Selected</span>' : ""}
              </span>
              <span class="breakdown-verdict">
                ${escapeHtml(score.prediction)}
                <b>${formatConfidence(score.confidence)}</b>
              </span>
            </div>
            <div class="breakdown-bar">
              <span class="${fake ? "is-fake" : "is-real"}" style="width: ${confidencePercent(
                score.confidence
              ).toFixed(1)}%"></span>
            </div>
          </div>`;
      })
      .join("");

    if (data.model_key === "ensemble") {
      breakdown.insertAdjacentHTML(
        "beforeend",
        `<p class="muted" style="margin:2px 0 0">Ensemble verdict = mean of the two model probabilities.</p>`
      );
    }

    if (data.disclaimer) disclaimerText.textContent = data.disclaimer;

    resultEmpty.hidden = true;
    resultContent.hidden = false;
    verdictBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /* ----------------------------- Submit ----------------------------- */
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = textarea.value;
    if (!validate(text)) return;

    const model = selectedModel();
    setLoading(true);
    hideError();

    try {
      const data = await apiFetch("/predict", {
        method: "POST",
        body: JSON.stringify({ text: text.trim(), model }),
      });
      renderResult(data);
      loadRecent();
      toast(
        `Classified as ${data.prediction} with ${formatConfidence(
          data.confidence
        )} confidence.`,
        "success"
      );
    } catch (error) {
      setLoading(false);
      showError(error.message || "Prediction failed. Please try again.");
      return;
    }
    setLoading(false);
  });

  anotherBtn.addEventListener("click", () => {
    resultContent.hidden = true;
    resultEmpty.hidden = false;
    textarea.value = "";
    updateCharCount();
    textarea.focus();
    resultEmpty.scrollIntoView({ behavior: "smooth", block: "center" });
    document.querySelectorAll("[data-sample]").forEach((b) => b.classList.remove("is-active"));
  });

  /* ----------------------------- Recent predictions ----------------------------- */
  function recentCard(row) {
    const fake = row.label === "FAKE";
    return `
      <article class="recent-item">
        <div class="recent-item-top">
          <span class="verdict-tag ${fake ? "fake" : "real"}">
            ${fake ? "FAKE" : "REAL"} NEWS
          </span>
          <span class="muted">${formatConfidence(row.confidence)}</span>
        </div>
        <p class="recent-preview">${escapeHtml(row.text_preview)}</p>
        <div class="recent-meta">
          <span>${escapeHtml(row.model || "—")}</span>
          <span>${formatTimestamp(row.created_at)}</span>
        </div>
      </article>`;
  }

  async function loadRecent() {
    if (!recentList) return;
    try {
      const data = await apiFetch("/api/history?limit=3");
      const rows = data.predictions || [];
      recentList.innerHTML = rows.length
        ? rows.map(recentCard).join("")
        : `<div class="recent-empty">No analyses yet. Your first prediction will appear here.</div>`;
    } catch {
      recentList.innerHTML = `<div class="recent-empty">Prediction history is currently unavailable.</div>`;
    }
  }

  loadRecent();

  /* Keyboard shortcut: "/" focuses the textarea */
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== textarea &&
        !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
      event.preventDefault();
      textarea.focus();
    }
  });
})();
