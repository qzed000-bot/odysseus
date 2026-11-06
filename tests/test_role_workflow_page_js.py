"""Behavior of the role-workflow page script under a fake browser.

Node loads the real static/js/role-workflow.js. The script is not copied.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
HAS_NODE = NODE is not None

PAGE = r"""
const fs = require("fs");
const calls = [];
const timers = [];
const elements = {};
function makeElement(id) {
  const node = {
    id: id,
    disabled: false,
    hidden: true,
    textContent: "",
    value: "",
    className: "",
    children: [],
    listeners: {},
    get firstChild() { return this.children[0] || null; },
    removeChild(child) {
      const index = this.children.indexOf(child);
      if (index >= 0) this.children.splice(index, 1);
      return child;
    },
    appendChild(child) { this.children.push(child); return child; },
    addEventListener(type, fn) { this.listeners[type] = fn; },
    reset() {},
  };
  return node;
}
global.document = {
  getElementById(id) {
    if (!elements[id]) elements[id] = makeElement(id);
    return elements[id];
  },
  createElement() { return makeElement(""); },
};
global.location = { hash: process.env.PAGE_HASH || "", pathname: "/static/role-workflow.html", search: "" };
global.history = {
  replaceState(_state, _title, url) {
    const text = String(url || "");
    const mark = text.indexOf("#");
    location.hash = mark === -1 ? "" : text.slice(mark);
  },
};
global.setTimeout = (fn) => { timers.push(fn); return timers.length; };
const script = fs.readFileSync(process.env.PAGE_JS, "utf8");
const projectId = "ab".repeat(16);
let project = {
  id: projectId,
  title: "Rain notes",
  brief: "Explain in one sentence how rain forms, for a general reader.",
  status: "active",
  endpoint_url: "http://127.0.0.1:11434/v1",
  model: "qwen2.5:7b-instruct",
  current_artifact: "",
  steps: [
    { position: 1, role: "writer", status: "running", instruction: "Writer", input_text: "Brief", output_text: "", error: "" },
    { position: 2, role: "critic", status: "pending", instruction: "Critic", input_text: "", output_text: "", error: "" },
    { position: 3, role: "organiser", status: "pending", instruction: "Organiser", input_text: "", output_text: "", error: "" },
  ],
};
let runPosts = 0;
let projectGets = 0;
const otherId = "cd".repeat(16);
if (process.env.PAGE_MODE === "late") project.steps[0].status = "pending";
global.fetch = (url, options) => {
  const method = (options && options.method) || "GET";
  calls.push(method + " " + url);
  if (method === "POST" && url.endsWith("/run")) {
    runPosts += 1;
    return Promise.resolve({ ok: false, status: 504, text: async () => JSON.stringify({ detail: "Request exceeded 45s timeout" }) });
  }
  if (method !== "GET") {
    return Promise.resolve({ ok: false, status: 500, text: async () => "{\"detail\":\"unexpected write\"}" });
  }
  if (url.endsWith("/status")) {
    return Promise.resolve({ ok: true, status: 200, text: async () => JSON.stringify({ enabled: true, models: ["qwen2.5:7b-instruct"] }) });
  }
  if (url.endsWith("/endpoints")) {
    return Promise.resolve({ ok: true, status: 200, text: async () => JSON.stringify({ endpoints: [{ id: "ep", name: "Private", base_url: "http://127.0.0.1:11434/v1" }] }) });
  }
  if (url.endsWith("/projects")) {
    return Promise.resolve({ ok: true, status: 200, text: async () => JSON.stringify({ projects: [
      { id: projectId, title: project.title, status: project.status },
      { id: otherId, title: "Other account", status: "active" },
    ] }) });
  }
  if (url.endsWith("/projects/" + projectId)) {
    projectGets += 1;
    if (process.env.PAGE_MODE === "forbidden") {
      return Promise.resolve({ ok: false, status: 404, text: async () => JSON.stringify({ detail: "Not found" }) });
    }
    if (process.env.PAGE_MODE === "late" && projectGets > 1) {
      const copy = JSON.parse(JSON.stringify(project));
      if (projectGets === 2) {
        copy.steps[0].status = "running";
        copy.steps[0].output_text = "";
      } else {
        copy.steps[0].status = "completed";
        copy.steps[0].output_text = "SAVED-LATE";
        copy.current_artifact = "SAVED-LATE";
      }
      return Promise.resolve({ ok: true, status: 200, text: async () => JSON.stringify(copy) });
    }
    return Promise.resolve({ ok: true, status: 200, text: async () => JSON.stringify(project) });
  }
  return Promise.resolve({ ok: false, status: 404, text: async () => JSON.stringify({ detail: "Not found" }) });
};
eval(script);
function textOf(id) { return elements[id] ? elements[id].textContent : ""; }
function outputText(node) {
  const root = node || elements.project;
  if (!root) return "";
  let text = root.textContent || "";
  (root.children || []).forEach((child) => { text += "\n" + outputText(child); });
  return text;
}
async function flush() {
  for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setImmediate(resolve));
}
(async () => {
  await flush();
  if (process.env.PAGE_MODE === "restore") {
    if (runPosts !== 0) throw new Error("restore posted a run");
    if (!outputText().includes("Rain notes")) throw new Error("selected project was not restored");
    if (elements["open-panel"].hidden) throw new Error("project panel stayed hidden");
  }
  if (process.env.PAGE_MODE === "forbidden") {
    if (!elements["open-panel"].hidden) throw new Error("forbidden project stayed visible");
    if (outputText().includes("Rain notes")) throw new Error("forbidden output leaked");
    if (!textOf("banner").includes("Not found")) throw new Error("missing forbidden error");
  }
  if (process.env.PAGE_MODE === "late") {
    if (elements.run.disabled) throw new Error("run started disabled");
    elements.run.listeners.click();
    await flush();
    if (runPosts !== 1) throw new Error("run was sent " + runPosts + " times");
    if (!outputText().includes("running")) throw new Error("lost response did not show the saved running state");
    if (!timers.length) throw new Error("no read scheduled after the lost response");
    timers.shift()();
    await flush();
    if (runPosts !== 1) throw new Error("poll sent another run");
    if (!outputText().includes("SAVED-LATE")) throw new Error("saved output was not shown");
    const writes = calls.filter((call) => !call.startsWith("GET "));
    if (writes.length !== 1 || !writes[0].includes("/run")) throw new Error("unexpected writes " + writes.join(","));
  }
  if (process.env.PAGE_MODE === "stale") {
    if (!outputText().includes("Rain notes")) throw new Error("setup did not show the saved project");
    const other = (elements.projects.children || []).find((node) => (node.textContent || "").includes("Other account"));
    if (!other) throw new Error("other project was not listed");
    other.listeners.click();
    await flush();
    if (!elements["open-panel"].hidden) throw new Error("stale id stayed open");
    if (outputText().includes("Rain notes")) throw new Error("stale id kept the previous output");
    if (!textOf("banner")) throw new Error("stale id had no error");
    const writes = calls.filter((call) => !call.startsWith("GET "));
    if (writes.length) throw new Error("stale read wrote " + writes.join(","));
  }
  console.log("PAGE_OK");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
"""


@pytest.mark.skipif(not HAS_NODE, reason="node binary not on PATH")
@pytest.mark.parametrize("mode,page_hash", [
    ("restore", "#" + "ab" * 16),
    ("forbidden", "#" + "ab" * 16),
    ("late", "#" + "ab" * 16),
    ("stale", "#" + "ab" * 16),
])
def test_role_workflow_page_reads_saved_state(mode, page_hash, tmp_path):
    script = tmp_path / "page-check.js"
    script.write_text(PAGE, encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "PAGE_JS": str(ROOT / "static" / "js" / "role-workflow.js"),
        "PAGE_MODE": mode,
        "PAGE_HASH": page_hash,
    })
    completed = subprocess.run(
        [NODE, str(script)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PAGE_OK" in completed.stdout
