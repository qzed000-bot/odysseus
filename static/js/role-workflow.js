"use strict";

var currentId = "";
var lastStatus = null;
var pollToken = 0;
var POLL_INTERVAL_MS = 2000;
var POLL_MAX_READS = 40;

function el(tag, className, text) {
  var node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function showError(message) {
  document.getElementById("banner").textContent = message || "";
}

function detailMessage(body, fallback) {
  if (!body) return fallback;
  var detail = body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return "Request was malformed";
  if (detail && typeof detail.detail === "string") return detail.detail;
  return fallback;
}

function api(path, options) {
  var opts = options || {};
  opts.credentials = "same-origin";
  opts.headers = { "Content-Type": "application/json", "Accept": "application/json" };
  return fetch(path, opts).then(function (response) {
    return response.text().then(function (text) {
      var body = null;
      if (text) {
        try {
          body = JSON.parse(text);
        } catch (err) {
          body = { detail: "Response was not JSON" };
        }
      }
      if (!response.ok) {
        var error = new Error(detailMessage(body, "Request failed"));
        error.status = response.status;
        throw error;
      }
      return body;
    });
  });
}

function selectedProjectId() {
  var raw = String(location.hash || "").replace(/^#/, "");
  return /^[0-9a-f]{32}$/.test(raw) ? raw : "";
}

function rememberProject(id) {
  if (!/^[0-9a-f]{32}$/.test(id)) return;
  if (selectedProjectId() === id) return;
  history.replaceState(null, "", location.pathname + location.search + "#" + id);
}

function stopWatch() {
  pollToken += 1;
}

function denied(error) {
  var status = error && error.status;
  return status === 401 || status === 403 || status === 404;
}

function stopAfterReadError(error) {
  if (denied(error)) {
    clearSelection();
    showError((error && error.message) || "This project is not available.");
    return;
  }
  stopWatch();
  showError((error && error.message) || "The saved project could not be read. Open it again. Do not press Run next step again.");
}

function clearSelection() {
  stopWatch();
  currentId = "";
  var panel = document.getElementById("open-panel");
  var root = document.getElementById("project");
  if (panel) panel.hidden = true;
  if (root) clear(root);
  if (location.hash) {
    history.replaceState(null, "", location.pathname + location.search);
  }
  var run = document.getElementById("run");
  if (run) run.disabled = false;
}

function stepIsRunning(project) {
  return (project.steps || []).some(function (step) { return step.status === "running"; });
}

function renderList(projects) {
  var list = document.getElementById("projects");
  clear(list);
  if (!projects.length) {
    list.appendChild(el("p", "muted", "No projects yet."));
    return;
  }
  projects.forEach(function (project) {
    var button = el("button", "project-button", project.title + " — " + project.status);
    button.type = "button";
    button.addEventListener("click", function () { openProject(project.id); });
    list.appendChild(button);
  });
}

function renderProject(project) {
  currentId = project.id;
  rememberProject(project.id);
  var root = document.getElementById("project");
  clear(root);
  document.getElementById("open-panel").hidden = false;
  document.getElementById("run").disabled = stepIsRunning(project);
  root.appendChild(el("h2", null, project.title));
  root.appendChild(el("p", null, "Project status: " + project.status));
  root.appendChild(el("p", "muted", project.endpoint_url + " · " + project.model));
  root.appendChild(el("h3", null, "Brief"));
  root.appendChild(el("pre", null, project.brief));
  root.appendChild(el("h3", null, "Current artifact"));
  root.appendChild(el("pre", null, project.current_artifact || ""));
  (project.steps || []).forEach(function (step) {
    var card = el("section", null, null);
    card.appendChild(el("h3", null, step.position + ". " + step.role + " (" + step.status + ")"));
    card.appendChild(el("h4", null, "Instruction"));
    card.appendChild(el("pre", null, step.instruction || ""));
    card.appendChild(el("h4", null, "Input the server saved for this step"));
    card.appendChild(el("pre", null, step.input_text || ""));
    card.appendChild(el("h4", null, "Saved output"));
    card.appendChild(el("pre", null, step.output_text || ""));
    if (step.error) card.appendChild(el("p", "error", step.error));
    root.appendChild(card);
  });
}

function loadProjects() {
  return api("/api/role-workflow/projects").then(function (body) {
    renderList(body.projects || []);
  });
}

function watchProject(id, token, readsSoFar) {
  if (token !== pollToken || currentId !== id) return;
  if (readsSoFar >= POLL_MAX_READS) {
    showError("This step is still marked running on the server. Open the project again to read the saved result. Do not press Run next step again.");
    return;
  }
  setTimeout(function () {
    if (token !== pollToken || currentId !== id) return;
    api("/api/role-workflow/projects/" + encodeURIComponent(id)).then(function (project) {
      if (token !== pollToken || !project || project.id !== id) return;
      renderProject(project);
      if (!stepIsRunning(project)) return;
      watchProject(id, token, readsSoFar + 1);
    }).catch(function (error) {
      if (token !== pollToken) return;
      stopAfterReadError(error);
    });
  }, POLL_INTERVAL_MS);
}

function watchIfRunning(project) {
  if (!project || !stepIsRunning(project)) {
    stopWatch();
    return;
  }
  var token = ++pollToken;
  watchProject(project.id, token, 1);
}

function openProject(id) {
  stopWatch();
  showError("");
  currentId = "";
  document.getElementById("open-panel").hidden = true;
  clear(document.getElementById("project"));
  return api("/api/role-workflow/projects/" + encodeURIComponent(id)).then(function (project) {
    if (!project || project.id !== id) {
      clearSelection();
      showError("This project is not available.");
      return null;
    }
    renderProject(project);
    watchIfRunning(project);
    return project;
  }).catch(function (error) {
    if (denied(error)) clearSelection();
    showError(error.message || "This project is not available.");
    return null;
  });
}

function fillSelect(select, options, selected) {
  clear(select);
  options.forEach(function (option) {
    var node = el("option", null, option.label);
    node.value = option.value;
    if (option.value === selected) node.selected = true;
    select.appendChild(node);
  });
}

function loadForm(status) {
  fillSelect(
    document.getElementById("model"),
    (status.models || []).map(function (model) { return { value: model, label: model }; })
  );
  return api("/api/role-workflow/endpoints").then(function (body) {
    var endpoints = body.endpoints || [];
    if (!endpoints.length) {
      fillSelect(document.getElementById("endpoint"), [{
        value: "",
        label: "No allowed loopback endpoint is saved",
      }]);
      document.getElementById("create").disabled = true;
      return;
    }
    document.getElementById("create").disabled = false;
    fillSelect(
      document.getElementById("endpoint"),
      endpoints.map(function (endpoint) {
        return { value: endpoint.id, label: endpoint.name + " — " + endpoint.base_url };
      })
    );
  });
}

document.getElementById("connect-local").addEventListener("click", function () {
  showError("");
  api("/api/role-workflow/connection", { method: "POST", body: "{}" })
    .then(function () {
      if (!lastStatus) return null;
      return loadForm(lastStatus);
    })
    .catch(function (error) { showError(error.message); });
});

document.getElementById("create-form").addEventListener("submit", function (event) {
  event.preventDefault();
  showError("");
  var payload = {
    title: document.getElementById("title").value,
    brief: document.getElementById("brief").value,
    endpoint_id: document.getElementById("endpoint").value,
    model: document.getElementById("model").value,
  };
  api("/api/role-workflow/projects", { method: "POST", body: JSON.stringify(payload) })
    .then(function (project) {
      document.getElementById("create-form").reset();
      renderProject(project);
      return loadProjects();
    })
    .catch(function (error) { showError(error.message); });
});

document.getElementById("run").addEventListener("click", function () {
  if (!currentId || document.getElementById("run").disabled) return;
  var id = currentId;
  var token = ++pollToken;
  showError("");
  document.getElementById("run").disabled = true;
  api("/api/role-workflow/projects/" + encodeURIComponent(id) + "/run", {
    method: "POST",
    body: "{}",
  }).then(function (project) {
    if (token !== pollToken) return;
    renderProject(project);
    watchIfRunning(project);
    return loadProjects();
  }).catch(function (error) {
    if (token !== pollToken) return;
    showError("The run request ended before the browser received the result (" + (error.message || "request failed") + "). Checking the saved project. This does not start another run.");
    api("/api/role-workflow/projects/" + encodeURIComponent(id)).then(function (project) {
      if (token !== pollToken || !project || project.id !== id) return;
      renderProject(project);
      if (stepIsRunning(project)) watchProject(id, token, 1);
    }).catch(function (readError) {
      if (token !== pollToken) return;
      stopAfterReadError(readError);
    });
  });
});

document.getElementById("recover").addEventListener("click", function () {
  if (!currentId) return;
  showError("");
  api("/api/role-workflow/projects/" + encodeURIComponent(currentId) + "/recover", {
    method: "POST",
    body: "{}",
  }).then(function (project) {
    renderProject(project);
    watchIfRunning(project);
  }).catch(function (error) { showError(error.message); });
});

document.getElementById("cancel").addEventListener("click", function () {
  if (!currentId) return;
  showError("");
  api("/api/role-workflow/projects/" + encodeURIComponent(currentId) + "/cancel", {
    method: "POST",
    body: "{}",
  }).then(function () {
    return openProject(currentId);
  }).catch(function (error) { showError(error.message); });
});

api("/api/role-workflow/status").then(function (status) {
  lastStatus = status;
  if (!status.enabled) {
    showError("This workflow is off. It stays off until ODYSSEUS_ROLE_WORKFLOW is exactly 1 and the local Ollama context guard is 2048.");
    document.getElementById("create").disabled = true;
    document.getElementById("connect-local").disabled = true;
    return;
  }
  return loadForm(status).then(loadProjects).then(function () {
    var selected = selectedProjectId();
    if (selected) return openProject(selected);
    return null;
  });
}).catch(function (error) {
  showError(error.message);
  document.getElementById("create").disabled = true;
});
