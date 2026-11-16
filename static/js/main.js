/* ==========================================================================
   FAKE NEWS DETECTOR — shared front-end utilities
   ========================================================================== */
(function () {
  "use strict";

  document.documentElement.classList.add("js");

  /* ---------------- Navigation ---------------- */
  const header = document.getElementById("siteHeader");
  const toggle = document.getElementById("navToggle");
  const navLinks = document.getElementById("navLinks");

  const closeNav = () => {
    if (!navLinks) return;
    navLinks.classList.remove("is-open");
    if (toggle) {
      toggle.setAttribute("aria-expanded", "false");
      toggle.setAttribute("aria-label", "Open menu");
    }
  };

  if (toggle && navLinks) {
    toggle.addEventListener("click", () => {
      const open = navLinks.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    });
    navLinks.querySelectorAll("a").forEach((link) =>
      link.addEventListener("click", closeNav)
    );
    document.addEventListener("click", (event) => {
      if (!navLinks.contains(event.target) && !toggle.contains(event.target)) closeNav();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeNav();
    });
  }

  const onScroll = () => {
    if (header) header.classList.toggle("is-scrolled", window.scrollY > 8);
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------------- Scroll reveal ---------------- */
  const revealItems = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && revealItems.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
    );
    revealItems.forEach((item) => observer.observe(item));
  } else {
    revealItems.forEach((item) => item.classList.add("is-visible"));
  }

  /* ---------------- Toasts ---------------- */
  const stack = document.getElementById("toastStack");
  const ICONS = {
    success: "#i-check",
    error: "#i-alert",
    info: "#i-info",
  };

  function toast(message, type = "info", duration = 5200) {
    if (!stack) return;
    const node = document.createElement("div");
    node.className = `toast is-${type}`;
    node.setAttribute("role", "status");
    node.innerHTML = `
      <svg class="icon"><use href="${ICONS[type] || ICONS.info}"/></svg>
      <span></span>`;
    node.querySelector("span").textContent = message;
    stack.appendChild(node);
    const remove = () => {
      node.classList.add("is-leaving");
      setTimeout(() => node.remove(), 260);
    };
    setTimeout(remove, duration);
    node.addEventListener("click", remove);
  }

  /* ---------------- API helper ---------------- */
  async function apiFetch(url, options = {}) {
    const response = await fetch(url, {
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
      ...options,
    });

    let data = null;
    const text = await response.text();
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { error: "The server returned an unexpected response." };
      }
    }

    if (!response.ok) {
      const message =
        (data && (data.error || data.message)) ||
        `Request failed with status ${response.status}.`;
      const error = new Error(message);
      error.status = response.status;
      error.data = data;
      throw error;
    }
    return data;
  }

  /* ---------------- Formatting helpers ---------------- */
  const numberFormat = new Intl.NumberFormat("en-US");

  function formatNumber(value) {
    return numberFormat.format(Number(value) || 0);
  }

  function formatPercent(value, digits = 1) {
    return `${(Number(value) * 100).toFixed(digits)}%`;
  }

  /**
   * Confidence values are model certainty, not factual truth. Rounding a
   * probability of 0.99996 would print "100.0%", which implies infallibility,
   * so cap the displayed percentage at 99.9%.
   */
  const MAX_DISPLAY_PERCENT = 99.9;

  function formatConfidence(value, digits = 1) {
    const percent = Number(value) * 100;
    if (!Number.isFinite(percent)) return "—";
    const capped = Math.min(percent, MAX_DISPLAY_PERCENT);
    return `${capped.toFixed(digits)}%`;
  }

  function confidencePercent(value) {
    const percent = Number(value) * 100;
    if (!Number.isFinite(percent)) return 0;
    return Math.min(percent, MAX_DISPLAY_PERCENT);
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatTimestamp(value) {
    if (!value) return "—";
    const normalized = String(value).includes("T")
      ? String(value)
      : String(value).replace(" ", "T");
    const date = new Date(normalized);
    if (Number.isNaN(date.getTime())) return String(value);
    const now = new Date();
    const diffMs = now - date;
    const minutes = Math.round(diffMs / 60000);
    if (minutes >= 0 && minutes < 1) return "just now";
    if (minutes >= 1 && minutes < 60) return `${minutes} min ago`;
    const hours = Math.round(minutes / 60);
    if (hours >= 1 && hours < 24) return `${hours} h ago`;
    return date.toLocaleString(undefined, {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  window.FND = {
    toast,
    apiFetch,
    formatNumber,
    formatPercent,
    formatConfidence,
    confidencePercent,
    escapeHtml,
    formatTimestamp,
    closeNav,
  };
})();
