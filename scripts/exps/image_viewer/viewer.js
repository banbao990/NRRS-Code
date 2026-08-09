(function () {
  "use strict";

  var MIN_SCALE = 1;
  var MAX_SCALE = 16;
  var MAX_DETAIL_SCALE = 5;
  var CROP_SCALE = 4;
  var ZOOM_SENSITIVITY = 0.0018;

  function ready(callback) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", callback, { once: true });
    } else {
      callback();
    }
  }

  function asFiniteNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    var number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function normalizeData(raw) {
    raw = raw && typeof raw === "object" ? raw : {};
    var sourceMethods = Array.isArray(raw.methods) ? raw.methods : [];
    var sourceCrops = Array.isArray(raw.crops) ? raw.crops : [];
    var methods = sourceMethods.map(function (entry, index) {
      entry = entry && typeof entry === "object" ? entry : { id: String(entry) };
      var id = String(entry.id !== undefined ? entry.id : "method-" + (index + 1));
      return {
        id: id,
        title: String(entry.title !== undefined ? entry.title : id),
        image: entry.image === undefined || entry.image === null ? "" : String(entry.image),
        relmse: asFiniteNumber(entry.relmse),
        rayeff: asFiniteNumber(entry.rayeff),
        isReference: entry.isReference === true || id === "reference"
      };
    });
    var crops = sourceCrops
      .map(function (entry, index) {
        entry = entry && typeof entry === "object" ? entry : {};
        var top = asFiniteNumber(entry.top);
        var left = asFiniteNumber(entry.left);
        var height = asFiniteNumber(entry.height);
        var width = asFiniteNumber(entry.width);
        if (
          top === null ||
          left === null ||
          height === null ||
          width === null ||
          top < 0 ||
          left < 0 ||
          height <= 0 ||
          width <= 0
        ) {
          return null;
        }
        return {
          title: String(entry.title !== undefined ? entry.title : "Crop " + (index + 1)),
          top: top,
          left: left,
          height: height,
          width: width
        };
      })
      .filter(function (entry) {
        return entry !== null;
      });

    return {
      scene: raw.scene === undefined || raw.scene === null ? "" : String(raw.scene),
      title: raw.title === undefined || raw.title === null ? "Image comparison" : String(raw.title),
      crops: crops,
      methods: methods
    };
  }

  function requiredElement(id) {
    var element = document.getElementById(id);
    if (!element) {
      throw new Error("Image viewer is missing required element #" + id + ".");
    }
    return element;
  }

  function formatNumber(value) {
    if (value === null || !Number.isFinite(value)) return "—";
    var magnitude = Math.abs(value);
    var digits = magnitude >= 100 ? 2 : magnitude >= 10 ? 3 : 4;
    return value.toFixed(digits).replace(/(\.[0-9]*?[1-9])0+$|\.0+$/, "$1");
  }

  function formatRelMSE(value) {
    return value === null ? "—" : formatNumber(value * 1000);
  }

  function setText(element, value) {
    element.textContent = value;
  }

  function createElement(tag, className, text) {
    var element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function isEditableTarget(target) {
    if (!(target instanceof Element)) return false;
    return Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
  }

  ready(function initializeViewer() {
    var data = normalizeData(window.VIEWER_DATA);
    var elements;

    document.title = data.scene ? data.title + " · " + data.scene : data.title;
    document.querySelectorAll("[data-viewer-title]").forEach(function (element) {
      setText(element, data.title);
    });
    document.querySelectorAll("[data-viewer-scene]").forEach(function (element) {
      setText(element, data.scene);
    });

    try {
      elements = {
        tabs: requiredElement("method-tabs"),
        stage: requiredElement("image-stage"),
        image: requiredElement("main-image"),
        title: requiredElement("method-title"),
        relmse: requiredElement("metric-relmse"),
        rayeff: requiredElement("metric-rayeff"),
        zoom: requiredElement("zoom-readout"),
        reset: requiredElement("reset-view"),
        cropActions: requiredElement("crop-actions"),
        thumbnails: requiredElement("thumbnail-strip"),
        metricsBody: requiredElement("metrics-body"),
        detailPanel: requiredElement("detail-panel"),
        detailStrip: requiredElement("detail-strip"),
        detailPosition: requiredElement("detail-position")
      };
    } catch (error) {
      console.error(error);
      return;
    }

    var state = {
      index: -1,
      scale: 1,
      x: 0,
      y: 0,
      pointerId: null,
      dragStartX: 0,
      dragStartY: 0,
      originX: 0,
      originY: 0,
      focusU: 0.5,
      focusV: 0.5,
      presetIndex: -1,
      presetU: 0.5,
      presetV: 0.5,
      loadToken: 0
    };

    var tabButtons = [];
    var cropButtons = [];
    var thumbnailButtons = [];
    var metricRows = [];
    var detailCards = [];
    var detailFrames = [];
    var detailSources = [];
    var errorMessage = null;

    elements.image.draggable = false;
    elements.image.decoding = "async";

    function removeError() {
      if (errorMessage) {
        errorMessage.remove();
        errorMessage = null;
      }
    }

    function showError(message) {
      removeError();
      errorMessage = createElement("div", "viewer-error", message);
      errorMessage.setAttribute("role", "status");
      elements.stage.appendChild(errorMessage);
    }

    function updateCropButtons() {
      cropButtons.forEach(function (button, index) {
        var active = index === state.presetIndex;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", active ? "true" : "false");
      });
    }

    function clearCropPreset() {
      state.presetIndex = -1;
      updateCropButtons();
    }

    function positionCropPreset() {
      if (state.presetIndex < 0 || state.presetIndex >= data.crops.length) return false;

      var sourceWidth = elements.image.naturalWidth;
      var sourceHeight = elements.image.naturalHeight;
      var baseWidth = elements.image.offsetWidth;
      var baseHeight = elements.image.offsetHeight;
      if (!sourceWidth || !sourceHeight || !baseWidth || !baseHeight) return false;

      var crop = data.crops[state.presetIndex];
      state.presetU = Math.min(1, Math.max(0, (crop.left + crop.width / 2) / sourceWidth));
      state.presetV = Math.min(1, Math.max(0, (crop.top + crop.height / 2) / sourceHeight));
      state.focusU = state.presetU;
      state.focusV = state.presetV;
      state.scale = CROP_SCALE;
      state.x = (0.5 - state.presetU) * baseWidth * state.scale;
      state.y = (0.5 - state.presetV) * baseHeight * state.scale;
      return true;
    }

    function clampPan() {
      if (state.scale <= 1) {
        state.x = 0;
        state.y = 0;
        return;
      }

      var baseWidth = elements.image.offsetWidth;
      var baseHeight = elements.image.offsetHeight;
      var stageWidth = elements.stage.clientWidth;
      var stageHeight = elements.stage.clientHeight;
      var limitX = Math.max(0, (baseWidth * state.scale - stageWidth) / 2);
      var limitY = Math.max(0, (baseHeight * state.scale - stageHeight) / 2);

      state.x = Math.min(limitX, Math.max(-limitX, state.x));
      state.y = Math.min(limitY, Math.max(-limitY, state.y));
    }

    function renderDetails() {
      var zoomed = state.scale > 1.001 && detailFrames.length > 0;
      elements.detailPanel.hidden = !zoomed;
      if (!zoomed) return;

      var detailScale = Math.min(MAX_DETAIL_SCALE, Math.max(1, state.scale));
      detailFrames.forEach(function (frame, index) {
        var source = detailSources[index];
        var sourceWidth = source && source.width ? source.width : 1280;
        var sourceHeight = source && source.height ? source.height : 720;
        var backgroundWidth = sourceWidth * detailScale;
        var backgroundHeight = sourceHeight * detailScale;
        var positionX = frame.clientWidth / 2 - state.focusU * backgroundWidth;
        var positionY = frame.clientHeight / 2 - state.focusV * backgroundHeight;

        frame.style.backgroundSize =
          backgroundWidth.toFixed(2) + "px " + backgroundHeight.toFixed(2) + "px";
        frame.style.backgroundPosition =
          positionX.toFixed(2) + "px " + positionY.toFixed(2) + "px";
      });

      var activeSource = detailSources[state.index];
      var width = activeSource && activeSource.width ? activeSource.width : 1280;
      var height = activeSource && activeSource.height ? activeSource.height : 720;
      var pixelX = Math.round(state.focusU * Math.max(0, width - 1));
      var pixelY = Math.round(state.focusV * Math.max(0, height - 1));
      setText(
        elements.detailPosition,
        "Focus " + pixelX + " × " + pixelY + " px · " + Math.round(detailScale * 100) + "%"
      );
    }

    function setDetailFocusFromClient(clientX, clientY) {
      var baseWidth = elements.image.offsetWidth;
      var baseHeight = elements.image.offsetHeight;
      if (!baseWidth || !baseHeight) return;

      var rect = elements.stage.getBoundingClientRect();
      var visualWidth = baseWidth * state.scale;
      var visualHeight = baseHeight * state.scale;
      var imageLeft = rect.left + (rect.width - visualWidth) / 2 + state.x;
      var imageTop = rect.top + (rect.height - visualHeight) / 2 + state.y;

      state.focusU = Math.min(1, Math.max(0, (clientX - imageLeft) / visualWidth));
      state.focusV = Math.min(1, Math.max(0, (clientY - imageTop) / visualHeight));
      renderDetails();
    }

    function setDetailFocusToStageCenter() {
      var rect = elements.stage.getBoundingClientRect();
      setDetailFocusFromClient(rect.left + rect.width / 2, rect.top + rect.height / 2);
    }

    function renderTransform() {
      if (!positionCropPreset()) clampPan();
      elements.image.style.transform =
        "translate3d(" + state.x.toFixed(2) + "px," + state.y.toFixed(2) + "px,0) " +
        "scale(" + state.scale.toFixed(5) + ")";
      setText(elements.zoom, Math.round(state.scale * 100) + "%");
      elements.stage.classList.toggle("is-zoomed", state.scale > 1.001);
      renderDetails();
    }

    function resetView() {
      clearCropPreset();
      state.scale = 1;
      state.x = 0;
      state.y = 0;
      state.focusU = 0.5;
      state.focusV = 0.5;
      renderTransform();
    }

    function focusCrop(index) {
      if (index < 0 || index >= data.crops.length) return;
      state.presetIndex = index;
      updateCropButtons();
      renderTransform();
      setDetailFocusToStageCenter();
    }

    function setActiveState(collection, activeIndex, attribute) {
      collection.forEach(function (element, index) {
        if (!element) return;
        var active = index === activeIndex;
        element.classList.toggle("is-active", active);
        element.setAttribute(attribute, active ? "true" : "false");
        element.tabIndex = active ? 0 : -1;
      });
    }

    function setMethod(index, options) {
      if (!data.methods.length) return;
      options = options || {};
      var length = data.methods.length;
      index = ((index % length) + length) % length;
      if (index === state.index && !options.force) return;

      state.index = index;
      var method = data.methods[index];
      var token = ++state.loadToken;

      setText(elements.title, method.title);
      setText(elements.relmse, formatRelMSE(method.relmse));
      setText(elements.rayeff, formatNumber(method.rayeff));
      elements.image.alt = data.title + " — " + method.title;
      setActiveState(tabButtons, index, "aria-selected");
      setActiveState(thumbnailButtons, index, "aria-current");
      setActiveState(metricRows, index, "aria-current");
      setActiveState(detailCards, index, "aria-current");
      removeError();

      if (!method.image) {
        elements.image.removeAttribute("src");
        elements.stage.classList.remove("is-loading");
        showError("No PNG path is available for " + method.title + ".");
        return;
      }

      elements.stage.classList.add("is-loading");
      elements.image.onload = function () {
        if (token !== state.loadToken) return;
        elements.stage.classList.remove("is-loading");
        renderTransform();
      };
      elements.image.onerror = function () {
        if (token !== state.loadToken) return;
        elements.stage.classList.remove("is-loading");
        showError("Could not load the PNG for " + method.title + ".");
      };

      // Assign the URL verbatim. In particular, do not decode percent-encoded paths:
      // encoded names are valid in both file:// pages and ordinary static hosting.
      elements.image.src = method.image;
      if (elements.image.complete && elements.image.naturalWidth > 0) {
        elements.stage.classList.remove("is-loading");
        renderTransform();
      }

      var activeTab = tabButtons[index];
      if (activeTab && options.ensureVisible) {
        activeTab.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
      }
    }

    function makeTab(method, index) {
      var button = createElement("button", "method-tab");
      if (method.isReference) button.classList.add("is-reference");
      button.type = "button";
      button.setAttribute("role", "tab");
      button.setAttribute("aria-label", "Show " + method.title);
      button.dataset.methodId = method.id;

      var keyLabel = index < 9 ? String(index + 1) : index === 9 ? "0" : "";
      if (keyLabel) button.appendChild(createElement("span", "method-tab-index", keyLabel));
      button.appendChild(createElement("span", "method-tab-label", method.title));
      button.addEventListener("click", function () {
        setMethod(index);
      });
      button.addEventListener("keydown", function (event) {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        event.stopPropagation();
        var offset = event.key === "ArrowLeft" ? -1 : 1;
        setMethod(index + offset, { ensureVisible: true });
        tabButtons[state.index].focus();
      });
      return button;
    }

    function makeThumbnail(method, index) {
      var button = createElement("button", "thumbnail-button");
      if (method.isReference) button.classList.add("is-reference");
      button.type = "button";
      button.setAttribute("aria-label", "Show " + method.title);
      button.dataset.methodId = method.id;

      var frame = createElement("span", "thumbnail-frame");
      var image = createElement("img", "thumbnail-image");
      image.alt = "";
      image.loading = "lazy";
      image.decoding = "async";
      image.draggable = false;
      if (method.image) image.src = method.image;
      frame.appendChild(image);
      button.appendChild(frame);
      button.appendChild(createElement("span", "thumbnail-label", method.title));
      button.addEventListener("click", function () {
        setMethod(index, { ensureVisible: true });
      });
      return button;
    }

    function makeDetail(method, index) {
      var button = createElement("button", "detail-card");
      button.type = "button";
      button.dataset.methodId = method.id;
      button.setAttribute("aria-label", "Show " + method.title);
      if (method.isReference) button.classList.add("is-reference");

      var frame = createElement("span", "detail-frame");
      frame.setAttribute("role", "img");
      frame.setAttribute("aria-label", "Synchronized detail for " + method.title);
      if (method.image) {
        // JSON quoting produces a valid CSS string while preserving encoded
        // method filenames such as 1%230%40nrrs%2B.png.
        frame.style.backgroundImage = "url(" + JSON.stringify(method.image) + ")";
      }
      button.appendChild(frame);
      button.appendChild(createElement("span", "detail-title", method.title));
      button.addEventListener("click", function () {
        setMethod(index, { ensureVisible: true });
      });

      var source = { width: 1280, height: 720 };
      if (method.image) {
        var preload = new Image();
        preload.decoding = "async";
        preload.onload = function () {
          source.width = preload.naturalWidth || source.width;
          source.height = preload.naturalHeight || source.height;
          renderDetails();
        };
        preload.src = method.image;
      }

      return { button: button, frame: frame, source: source };
    }

    function makeMetricRow(method, index) {
      var row = createElement("tr", "metrics-row");
      row.tabIndex = 0;
      row.dataset.methodId = method.id;
      row.setAttribute("aria-label", "Show " + method.title);
      row.appendChild(createElement("td", "metrics-method", method.title));
      row.appendChild(createElement("td", "metrics-relmse", formatRelMSE(method.relmse)));
      row.appendChild(createElement("td", "metrics-rayeff", formatNumber(method.rayeff)));
      row.addEventListener("click", function () {
        setMethod(index, { ensureVisible: true });
      });
      row.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          setMethod(index, { ensureVisible: true });
        }
      });
      return row;
    }

    function buildCropControls() {
      elements.cropActions.replaceChildren();
      cropButtons.length = 0;
      elements.cropActions.hidden = data.crops.length === 0;

      data.crops.forEach(function (crop, index) {
        var button = createElement("button", "ghost-button crop-button", crop.title);
        button.type = "button";
        button.setAttribute("aria-label", "Center " + crop.title + " at 400% zoom");
        button.setAttribute("aria-pressed", "false");
        button.addEventListener("click", function () {
          focusCrop(index);
        });
        cropButtons.push(button);
        elements.cropActions.appendChild(button);
      });
    }

    function buildMethodControls() {
      elements.tabs.replaceChildren();
      elements.thumbnails.replaceChildren();
      elements.metricsBody.replaceChildren();
      elements.detailStrip.replaceChildren();
      elements.tabs.setAttribute("role", "tablist");
      elements.tabs.setAttribute("aria-label", "Rendering methods");

      data.methods.forEach(function (method, index) {
        var tab = makeTab(method, index);
        var thumbnail = makeThumbnail(method, index);
        var detail = makeDetail(method, index);
        var metricRow = method.isReference ? null : makeMetricRow(method, index);
        tabButtons.push(tab);
        thumbnailButtons.push(thumbnail);
        metricRows.push(metricRow);
        detailCards.push(detail.button);
        detailFrames.push(detail.frame);
        detailSources.push(detail.source);
        elements.tabs.appendChild(tab);
        elements.thumbnails.appendChild(thumbnail);
        elements.detailStrip.appendChild(detail.button);
        if (metricRow) elements.metricsBody.appendChild(metricRow);
      });
    }

    function zoomAt(clientX, clientY, nextScale) {
      nextScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, nextScale));
      if (Math.abs(nextScale - state.scale) < 0.0001) return;

      var rect = elements.stage.getBoundingClientRect();
      var pointX = clientX - rect.left - rect.width / 2;
      var pointY = clientY - rect.top - rect.height / 2;
      var ratio = nextScale / state.scale;
      state.x = pointX - (pointX - state.x) * ratio;
      state.y = pointY - (pointY - state.y) * ratio;
      state.scale = nextScale;
      renderTransform();
    }

    elements.stage.addEventListener(
      "wheel",
      function (event) {
        event.preventDefault();
        clearCropPreset();
        if (state.scale > 1.001) {
          setDetailFocusFromClient(event.clientX, event.clientY);
        } else {
          // The synchronized details should have a deterministic first frame.
          // Pointer tracking starts only after the detail panel is visible.
          state.focusU = 0.5;
          state.focusV = 0.5;
        }
        var factor = Math.exp(-event.deltaY * ZOOM_SENSITIVITY);
        zoomAt(event.clientX, event.clientY, state.scale * factor);
      },
      { passive: false }
    );

    elements.stage.addEventListener("pointerdown", function (event) {
      if (event.button !== 0 || state.scale <= 1) return;
      clearCropPreset();
      setDetailFocusFromClient(event.clientX, event.clientY);
      state.pointerId = event.pointerId;
      state.dragStartX = event.clientX;
      state.dragStartY = event.clientY;
      state.originX = state.x;
      state.originY = state.y;
      elements.stage.setPointerCapture(event.pointerId);
      elements.stage.classList.add("is-dragging");
      event.preventDefault();
    });

    elements.stage.addEventListener("pointermove", function (event) {
      if (event.pointerId === state.pointerId) {
        state.x = state.originX + event.clientX - state.dragStartX;
        state.y = state.originY + event.clientY - state.dragStartY;
        renderTransform();
      }
      if (state.scale > 1.001 && state.presetIndex < 0) {
        setDetailFocusFromClient(event.clientX, event.clientY);
      }
    });

    function finishDrag(event) {
      if (event.pointerId !== state.pointerId) return;
      state.pointerId = null;
      elements.stage.classList.remove("is-dragging");
      if (elements.stage.hasPointerCapture(event.pointerId)) {
        elements.stage.releasePointerCapture(event.pointerId);
      }
    }

    elements.stage.addEventListener("pointerup", finishDrag);
    elements.stage.addEventListener("pointercancel", finishDrag);
    elements.stage.addEventListener("dblclick", function (event) {
      event.preventDefault();
      resetView();
    });
    elements.reset.addEventListener("click", resetView);

    document.addEventListener("keydown", function (event) {
      if (event.altKey || event.ctrlKey || event.metaKey || isEditableTarget(event.target)) return;

      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        setMethod(state.index + (event.key === "ArrowLeft" ? -1 : 1), {
          ensureVisible: true
        });
        return;
      }

      if (event.key === "r" || event.key === "R") {
        event.preventDefault();
        resetView();
        return;
      }

      if (/^[0-9]$/.test(event.key)) {
        var requestedIndex = event.key === "0" ? 9 : Number(event.key) - 1;
        if (requestedIndex < data.methods.length) {
          event.preventDefault();
          setMethod(requestedIndex, { ensureVisible: true });
        }
      }
    });

    window.addEventListener("resize", renderTransform, { passive: true });

    buildCropControls();
    buildMethodControls();
    if (!data.methods.length) {
      setText(elements.title, "No methods available");
      setText(elements.relmse, "—");
      setText(elements.rayeff, "—");
      setText(elements.zoom, "100%");
      showError("VIEWER_DATA does not contain any methods.");
      return;
    }

    setMethod(0, { force: true });
  });
})();
