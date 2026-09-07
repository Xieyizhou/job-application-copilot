const ENDPOINT = "http://127.0.0.1:8765";
const setup = document.querySelector("#setup");
const tokenInput = document.querySelector("#token");
const importButton = document.querySelector("#import");
const status = document.querySelector("#status");
const APP_URL = "http://127.0.0.1:8501";

function setStatus(message, tone = "") {
  status.textContent = message;
  status.className = tone;
}

function extractPage() {
  const clean = (value) => String(value || "").replace(/\u00a0/g, " ").replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
  const textFromHtml = (value) => {
    const doc = new DOMParser().parseFromString(String(value || ""), "text/html");
    return clean(doc.body?.innerText || doc.body?.textContent || "");
  };
  const postings = [];
  const visit = (value) => {
    if (Array.isArray(value)) return value.forEach(visit);
    if (!value || typeof value !== "object") return;
    const types = Array.isArray(value["@type"]) ? value["@type"] : [value["@type"]];
    if (types.some((item) => String(item).toLowerCase() === "jobposting")) postings.push(value);
    Object.entries(value).forEach(([key, item]) => { if (key !== "@context") visit(item); });
  };
  document.querySelectorAll('script[type="application/ld+json"]').forEach((script) => {
    try { visit(JSON.parse(script.textContent || "")); } catch (_) { /* Ignore malformed site data. */ }
  });
  if (postings.length) {
    const posting = postings.sort((a, b) => String(b.description || "").length - String(a.description || "").length)[0];
    const organization = posting.hiringOrganization || {};
    const parts = [posting.description, posting.responsibilities, posting.qualifications, posting.skills, posting.experienceRequirements, posting.educationRequirements]
      .map(textFromHtml).filter(Boolean);
    return {
      url: location.href,
      title: clean(posting.title || document.title),
      company: clean(organization.name || ""),
      description: [...new Set(parts)].join("\n\n"),
      extractor: "browser_jobposting_json_ld"
    };
  }
  const selectors = [
    '[itemprop="description"]', '[data-testid*="job-description"]', '[data-testid*="jobDescription"]',
    '[data-automation-id="jobPostingDescription"]', '[data-ph-at-id="job-description-text"]',
    '#job-description', '.job-description', '.job_description', '.jobDescription',
    '.posting-description', '.description-content', 'main article', 'article', 'main'
  ];
  const candidates = selectors.flatMap((selector) => [...document.querySelectorAll(selector)])
    .map((element) => clean(element.innerText || element.textContent || ""))
    .filter((value) => value.split(/\s+/).length >= 50);
  const description = candidates.sort((a, b) => b.length - a.length)[0] || "";
  return {
    url: location.href,
    title: clean(document.querySelector("h1")?.textContent || document.title),
    company: clean(
      document.querySelector('[itemprop="hiringOrganization"]')?.textContent ||
      document.querySelector('[data-automation-id="jobPostingCompany"]')?.textContent ||
      document.querySelector('[data-ph-at-id="job-company"]')?.textContent ||
      document.querySelector('[data-testid*="company"]')?.textContent ||
      document.querySelector('.company-name, .companyName, .job-company, [class*="company-name"]')?.textContent || ""
    ),
    description,
    extractor: "browser_visible_job_container"
  };
}

async function currentToken() {
  const stored = await chrome.storage.local.get("connectionToken");
  return String(stored.connectionToken || "");
}

async function updateSetupVisibility(force = false) {
  setup.hidden = !force && Boolean(await currentToken());
}

document.querySelector("#save-token").addEventListener("click", async () => {
  const token = tokenInput.value.trim();
  if (!token) return setStatus("Paste the token shown in JobCopilot Settings.", "error");
  await chrome.storage.local.set({ connectionToken: token });
  tokenInput.value = "";
  setup.hidden = true;
  setStatus("Local connection saved.", "success");
});

document.querySelector("#change-token").addEventListener("click", () => updateSetupVisibility(true));
document.querySelector("#open-app").addEventListener("click", () => {
  chrome.tabs.create({ url: APP_URL });
});

importButton.addEventListener("click", async () => {
  const token = await currentToken();
  if (!token) {
    await updateSetupVisibility(true);
    return setStatus("Add the local token before importing.", "error");
  }
  importButton.disabled = true;
  setStatus("Reading and verifying the opened posting…", "pending");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id || !/^https?:/.test(tab.url || "")) throw new Error("Open a public job posting first.");
    const [{ result: payload }] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: extractPage });
    const response = await fetch(`${ENDPOINT}/v1/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-JobCopilot-Token": token },
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.message || "The capture was rejected.");
    setStatus("Full JD verified and saved. Open JobCopilot to review the new score.", "success");
  } catch (error) {
    const message = error instanceof TypeError
      ? "JobCopilot is not running locally, or the connection token is no longer current."
      : String(error.message || error);
    setStatus(message, "error");
  } finally {
    importButton.disabled = false;
  }
});

updateSetupVisibility();
