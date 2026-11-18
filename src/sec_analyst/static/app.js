// SEC Filing Analyst front end. Plain JavaScript, no build step, no external scripts.
// All text from the server is inserted with textContent (never innerHTML) to prevent XSS.
(function () {
  "use strict";

  var form = document.getElementById("ask-form");
  var question = document.getElementById("question");
  var button = document.getElementById("ask-button");
  var area = document.getElementById("answer-area");
  var charCount = document.getElementById("char-count");
  var pills = document.getElementById("company-pills");
  var themeToggle = document.getElementById("theme-toggle");
  var emptyState = document.getElementById("empty-state");

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  // ---- theme -------------------------------------------------------------------------
  function currentTheme() {
    var set = document.documentElement.dataset.theme;
    if (set) return set;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function updateThemeLabel() {
    themeToggle.setAttribute("aria-label", currentTheme() === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }
  themeToggle.addEventListener("click", function () {
    var next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* storage may be blocked */ }
    updateThemeLabel();
  });
  updateThemeLabel();

  // ---- company pills (from /filings) -------------------------------------------------
  function addPill(value, label, title) {
    var wrap = el("label", "pill");
    var input = el("input");
    input.type = "radio"; input.name = "ticker"; input.value = value;
    var span = el("span", null, label);
    if (title) wrap.title = title;
    wrap.appendChild(input); wrap.appendChild(span);
    pills.appendChild(wrap);
  }
  fetch("/filings").then(function (r) { return r.ok ? r.json() : []; }).then(function (items) {
    items.forEach(function (f) {
      addPill(f.ticker, f.ticker, f.company + (f.fiscal_year ? ", 10-K fiscal " + f.fiscal_year : ""));
    });
  }).catch(function () { /* the page still works with "All companies" */ });

  // ---- form ----------------------------------------------------------------------------
  question.addEventListener("input", function () { charCount.textContent = question.value.length + " / 500"; });
  question.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });
  Array.prototype.forEach.call(document.querySelectorAll(".example"), function (b) {
    b.addEventListener("click", function () {
      question.value = b.dataset.q;
      question.dispatchEvent(new Event("input"));
      var radio = pills.querySelector('input[value="' + (b.dataset.t || "") + '"]');
      if (radio) radio.checked = true;
      form.requestSubmit();
    });
  });

  function selected(name) {
    var checked = form.querySelector('input[name="' + name + '"]:checked');
    return checked ? checked.value : "";
  }

  function setLoading(on) {
    button.disabled = on;
    button.classList.toggle("is-loading", on);
    button.querySelector(".button-label").textContent = on ? "Thinking" : "Ask";
    area.setAttribute("aria-busy", on ? "true" : "false");
  }

  function showSkeleton(mode) {
    area.replaceChildren();
    var sk = el("div", "skeleton");
    for (var i = 0; i < 4; i++) sk.appendChild(el("div", "skeleton-line"));
    area.appendChild(sk);
    area.appendChild(el("p", "loading-text", mode === "agent"
      ? "The agent is searching the filings and calling tools. This can take 10 to 30 seconds."
      : "Searching the filings and drafting an answer..."));
  }

  function showError(message) {
    area.replaceChildren(el("div", "alert alert-error", message));
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var q = question.value.trim();
    if (q.length < 3) {
      showError("Please type a question of at least 3 characters.");
      question.focus();
      return;
    }
    var mode = selected("mode") || "agent";
    var body = { question: q, mode: mode };
    var ticker = selected("ticker");
    if (ticker) body.ticker = ticker;

    setLoading(true);
    showSkeleton(mode);
    fetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (data) { return { ok: r.ok, status: r.status, data: data }; });
      })
      .then(function (res) {
        if (!res.ok) {
          var detail = res.data && res.data.detail;
          if (Array.isArray(detail)) detail = detail.map(function (d) { return d.msg; }).join("; ");
          showError(detail || ("Request failed (HTTP " + res.status + ")."));
          return;
        }
        renderAnswer(res.data);
      })
      .catch(function () { showError("Could not reach the server. Is it running?"); })
      .then(function () { setLoading(false); });
  });

  // ---- rendering -----------------------------------------------------------------------
  function normalize(s) {
    return s.replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/[–—]/g, "-").replace(/\s+/g, " ").toLowerCase();
  }

  // Show the passage with the quoted text highlighted (first part of an ellipsis quote).
  function renderPassage(passage, quote) {
    var pre = el("p", "passage");
    var part = (quote || "").split(/\.\.\.|…/)[0].trim().replace(/^["']|["']$/g, "");
    var flat = passage.replace(/\s+/g, " ");
    var idx = part.length >= 8 ? normalize(flat).indexOf(normalize(part)) : -1;
    if (idx < 0) { pre.textContent = flat; return pre; }
    pre.appendChild(document.createTextNode(flat.slice(0, idx)));
    pre.appendChild(el("mark", null, flat.slice(idx, idx + part.length)));
    pre.appendChild(document.createTextNode(flat.slice(idx + part.length)));
    return pre;
  }

  function renderAnswer(data) {
    area.replaceChildren();
    var badges = el("div", "badges");
    if (!data.found) badges.appendChild(el("span", "badge badge-neutral", "Not found in filings"));
    else if (data.grounded) badges.appendChild(el("span", "badge badge-success", "Citations verified"));
    else badges.appendChild(el("span", "badge badge-warning", "No verified citation"));
    badges.appendChild(el("span", "badge badge-neutral", data.mode === "agent" ? "Agent" : "Direct"));
    area.appendChild(badges);
    area.appendChild(el("p", "answer-text", data.answer));

    var cites = data.citations || [];
    if (cites.length) {
      area.appendChild(el("p", "section-label", "Sources"));
      var chips = el("div", "chips");
      var cards = el("div");
      cites.forEach(function (c, i) {
        var id = "quote-" + i;
        var chip = el("button", "chip" + (c.valid ? "" : " unverified"));
        chip.type = "button";
        chip.setAttribute("aria-expanded", "false");
        chip.setAttribute("aria-controls", id);
        chip.setAttribute("aria-label", "Source " + (i + 1) + ": " + (c.company || c.chunk_id) + (c.valid ? "" : " (quote not verified)"));
        chip.appendChild(el("span", "chip-num", String(i + 1)));
        chip.appendChild(el("span", null, (c.ticker || "?") + " · " + (c.section ? c.section.split(".")[0] : "unknown")));

        var card = el("div", "quote-card");
        card.id = id; card.hidden = true;
        var meta = el("p", "quote-meta");
        meta.appendChild(el("strong", null, (c.company || "Unknown source") + (c.fiscal_year ? " · FY" + c.fiscal_year : "")));
        meta.appendChild(document.createTextNode(c.section ? " · " + c.section : ""));
        card.appendChild(meta);
        card.appendChild(el("div", "quote-id", c.chunk_id + (c.valid ? "" : "  —  quote not found in this passage") +
          (c.corrected_from ? "  —  re-attributed (the model cited " + c.corrected_from + ")" : "")));
        card.appendChild(c.passage ? renderPassage(c.passage, c.quote) : el("p", "passage", c.quote));

        chip.addEventListener("click", function () {
          var open = chip.getAttribute("aria-expanded") === "true";
          chip.setAttribute("aria-expanded", open ? "false" : "true");
          card.hidden = open;
        });
        chips.appendChild(chip);
        cards.appendChild(card);
      });
      area.appendChild(chips);
      area.appendChild(cards);
    }

    var steps = (data.steps || []).filter(function (s) { return s.type === "tool"; });
    if (steps.length) {
      var details = el("details", "steps");
      details.appendChild(el("summary", null, "How the agent worked (" + steps.length + " tool call" + (steps.length > 1 ? "s" : "") + ")"));
      var ol = el("ol");
      steps.forEach(function (s) {
        var li = el("li");
        li.appendChild(el("code", null, s.tool + "(" + JSON.stringify(s.args) + ")"));
        li.appendChild(document.createTextNode(" → " + s.summary));
        ol.appendChild(li);
      });
      details.appendChild(ol);
      area.appendChild(details);
    }
    if (data.note && data.found) area.appendChild(el("div", "alert alert-warning", "Note: " + data.note));
    var secs = (data.latency_ms / 1000).toFixed(1);
    area.appendChild(el("p", "meta-line", "Model " + data.model + " · " + secs + " s"));
  }

  if (emptyState) emptyState.hidden = false;
})();
