"use strict";

var currentId = "";

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
  var root = document.getElementById("project");
  clear(root);
  document.getElementById("open-panel").hidden = false;
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

function openProject(id) {
  showError("");
  return api("/api/role-workflow/projects/" + encodeURIComponent(id)).then(function (project) {
    renderProject(project);
  }).catch(function (error) {
    showError(error.message);
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

var lastStatus = null;

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
  if (!currentId) return;
  showError("");
  api("/api/role-workflow/projects/" + encodeURIComponent(currentId) + "/run", {
    method: "POST",
    body: "{}",
  }).then(function (project) {
    renderProject(project);
    return loadProjects();
  }).catch(function (error) {
    showError(error.message);
    return openProject(currentId);
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
  return loadForm(status).then(loadProjects);
}).catch(function (error) {
  showError(error.message);
  document.getElementById("create").disabled = true;
});
