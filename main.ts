// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief ($MAKEFUN_PROMPT_FILE), probes every
// provider's catalog route in its own wire, builds one catalog, registers
// with the env, and answers the client's twenty requests — each with the
// best free model that meets the request's requirements, failing over to
// the next-best free model when a provider refuses.
//
// Routing, in one line: every free candidate that meets the requirements,
// best rating first, one provider call at a time, and a provider that
// refuses twice in a row is demoted below the rest for the next two
// requests (its budget may be spent, or its outage may have begun; a lone
// refusal is a failure-schedule hit and costs nothing). The quality table
// in the brief is the sealed truth for ratings and capabilities; a paid
// model is never a legal pick and is never chosen.
//
// The wires: an OpenAI client and an Anthropic client, each in JSON and in
// server-sent events, answered by an OpenAI, Anthropic or Gemini provider —
// all six combinations, each translated so the provider echoes exactly what
// the client sent (system prompt, user text, images, tools, stop sequences,
// max_tokens, JSON mode) and the answer comes back in the wire the client
// used, with the provider's text whole, its usage counts, and its stop
// reason mapped.
//
// Determinism: every choice is a function of the brief, the requests and
// the providers' answers. No wall clock, no unseeded randomness, one
// request at a time to the env, in order — so a replay leaves the same
// state export byte for byte.
//
// Run as `node main.ts` in a trial, or `node main.ts --list` to print the
// catalog as one JSON document and exit 0. Outside a trial (no brief, no
// env) --list prints {"providers":[]} and exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  byModelId,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let requestIndex = 0;

/**
 * What this server knows about one provider's health, learned only from
 * its answers: consecutive refusals, and the request index until which it
 * stays demoted. A provider that refuses twice running is tried last for
 * the next two requests, then tried in rating order again.
 */
interface ProviderHealth {
  consecutiveFailures: number;
  demotedUntil: number;
}
const health = new Map<string, ProviderHealth>();

function healthOf(providerId: string): ProviderHealth {
  let h = health.get(providerId);
  if (h === undefined) {
    h = { consecutiveFailures: 0, demotedUntil: -1 };
    health.set(providerId, h);
  }
  return h;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byModelId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  demoted: boolean;
}

/**
 * Every free model that meets one request's requirements, best rating
 * first. A provider whose wire cannot carry JSON mode is skipped for a
 * JSON-mode request from an OpenAI client (its echo would say the mode was
 * absent); an Anthropic client never carries JSON mode, so any wire serves
 * those. Providers demoted for consecutive refusals sort after the rest.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean, clientWire: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && clientWire === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const h = healthOf(probe.provider.id);
      out.push({ probe, model, rating: row?.rating_milli ?? 0, demoted: requestIndex <= h.demotedUntil });
    }
  }
  out.sort((a, b) => {
    if (a.demoted !== b.demoted) return a.demoted ? 1 : -1;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, index: number): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[index];
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether an OpenAI client request asks for JSON mode. */
function openAIJsonMode(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// ---------------------------------------------------------------------------
// Upstream calls: one provider, one wire, with failover.
// ---------------------------------------------------------------------------

/** The outcome of one provider call: its status and its whole body as text. */
interface UpstreamResult {
  ok: boolean;
  status: number;
  text: string;
}

/** Call one candidate provider with the request translated to its wire. */
async function callCandidate(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientWire === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstreamBody.stream_options = { include_usage: true };
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    upstreamBody = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`;
    upstreamBody = clientWire === 'openai' ? openAIToGemini(body, openAIJsonMode(body)) : anthropicToGemini(body);
    if (stream) url += '?alt=sse';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, text: (err as Error).message };
  }
  const text = await res.text();
  return { ok: res.ok, status: res.status, text };
}

/** Whether an upstream 2xx body is a well-formed answer on its wire (else fail over). */
function answerWellFormed(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, text: string): boolean {
  if (stream) {
    // A stream must carry at least one data frame that parses as JSON.
    for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
      if (frame.trim() === '') continue;
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') return true;
        try {
          if (isRecord(JSON.parse(payload))) return true;
        } catch {
          // Not JSON: keep looking.
        }
      }
    }
    return false;
  }
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value)) return false;
    if (wire === 'openai') return Array.isArray(value.choices) && value.choices.length > 0;
    if (wire === 'anthropic') return value.type === 'message' && Array.isArray(value.content);
    return Array.isArray(value.candidates) && value.candidates.length > 0;
  } catch {
    return false;
  }
}

/** Record one provider's answer in its health: a success clears refusals, a refusal counts toward demotion. */
function noteAnswer(providerId: string, ok: boolean): void {
  const h = healthOf(providerId);
  if (ok) {
    h.consecutiveFailures = 0;
    h.demotedUntil = -1;
    return;
  }
  h.consecutiveFailures += 1;
  if (h.consecutiveFailures >= 2) h.demotedUntil = requestIndex + 2;
}

/**
 * Try the candidates in order until one answers: each refusal is counted
 * and the next-best free model serves instead. Returns the answering
 * candidate and its whole upstream body, or null when none answered.
 */
async function serveWithFailover(candidates: Candidate[], clientWire: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<{ candidate: Candidate; result: UpstreamResult } | null> {
  for (const candidate of candidates) {
    const result = await callCandidate(candidate, clientWire, body, stream);
    const wellFormed = result.ok && answerWellFormed(candidate.probe.provider.wire, stream, result.text);
    noteAnswer(candidate.probe.provider.id, wellFormed);
    if (wellFormed) return { candidate, result };
    log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} answered HTTP ${result.status}`);
  }
  return null;
}

// ---------------------------------------------------------------------------
// Answering the client: translate the provider's answer into the client's wire.
// ---------------------------------------------------------------------------

/** Send one provider's JSON answer to the client, translated to its wire. */
function sendTranslatedJson(res: ServerResponse, clientWire: 'openai' | 'anthropic', providerWire: 'openai' | 'anthropic' | 'gemini', text: string, model: string): void {
  const parsed: unknown = JSON.parse(text);
  const body = isRecord(parsed) ? parsed : {};
  if (clientWire === 'openai') {
    if (providerWire === 'openai') sendJson(res, 200, body);
    else if (providerWire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResponse(body, model));
    else sendJson(res, 200, geminiToOpenAI(body, model));
    return;
  }
  if (providerWire === 'openai') sendJson(res, 200, openAIToAnthropic(body, model));
  else if (providerWire === 'anthropic') sendJson(res, 200, body);
  else sendJson(res, 200, geminiToAnthropic(body, model));
}

/** Send one provider's SSE answer to the client, translated to its wire. */
async function sendTranslatedStream(res: ServerResponse, clientWire: 'openai' | 'anthropic', providerWire: 'openai' | 'anthropic' | 'gemini', text: string, model: string): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (clientWire === 'openai' && providerWire === 'openai') {
    // Pass the provider's OpenAI stream through byte for byte.
    res.end(text);
    return;
  }
  if (clientWire === 'anthropic' && providerWire === 'anthropic') {
    res.end(text);
    return;
  }
  // Translate: read the provider's frames from the text just received.
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
  if (clientWire === 'openai' && providerWire === 'anthropic') {
    for await (const event of openAIEventsFromAnthropic(stream, model)) res.write(event);
  } else if (clientWire === 'openai' && providerWire === 'gemini') {
    for await (const event of openAIEventsFromGemini(stream, model)) res.write(event);
  } else if (clientWire === 'anthropic' && providerWire === 'openai') {
    for await (const event of anthropicEventsFromOpenAI(stream, model)) res.write(event);
  } else {
    for await (const event of anthropicEventsFromGemini(stream, model)) res.write(event);
  }
  res.end();
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const index = requestIndex;
  requestIndex += 1;
  const stream = body.stream === true;
  const candidates = candidatesFor(requirementsFor(body, index), openAIJsonMode(body), 'openai');
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  const served = await serveWithFailover(candidates, 'openai', body, stream);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'every free model that meets this request refused it');
  const providerWire = served.candidate.probe.provider.wire;
  const model = typeof body.model === 'string' && body.model !== '' ? body.model : served.candidate.model.id;
  if (stream) await sendTranslatedStream(res, 'openai', providerWire, served.result.text, model);
  else sendTranslatedJson(res, 'openai', providerWire, served.result.text, model);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const index = requestIndex;
  requestIndex += 1;
  const stream = body.stream === true;
  const candidates = candidatesFor(requirementsFor(body, index), false, 'anthropic');
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  const served = await serveWithFailover(candidates, 'anthropic', body, stream);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'every free model that meets this request refused it');
  const providerWire = served.candidate.probe.provider.wire;
  const model = typeof body.model === 'string' && body.model !== '' ? body.model : served.candidate.model.id;
  if (stream) await sendTranslatedStream(res, 'anthropic', providerWire, served.result.text, model);
  else sendTranslatedJson(res, 'anthropic', providerWire, served.result.text, model);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
