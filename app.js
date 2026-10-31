import { Wllama } from './vendor/wllama/index.js';

const MODEL_URL = new URL('model/bytenpc-50m-q8_0.gguf', location.href).href;
const WASM = { default: new URL('vendor/wllama/wllama.wasm', location.href).href };
const N_CTX = 1024, MAX_NEW = 60, MAX_INPUT = 400;
const PROMPT_CHAR_BUDGET = 3200;           // ~3.5 chars/token for this tokenizer: keeps prompt + reply under 1024 tokens

const PRESETS = [
  { label: 'Tavern: the innkeeper', setting: { name: 'The Rusty Tankard', category: 'Tavern',
      description: 'A smoky tavern at the edge of the village. Travelers share ale and rumors by the fire.',
      background: 'The innkeeper knows every story in the valley and sells a good stew.' },
    user: { name: 'traveler', persona: 'I am a tired traveler from the north. I seek food, a bed and news of the road.' },
    bot: { name: 'innkeeper', persona: 'I run this tavern. I am kind but I love gossip, and I never give credit.' } },
  { label: 'Village square: the elder with a quest', setting: { name: 'Village Square', category: 'Village',
      description: 'A muddy square with a well in the middle. The houses are shut and the people look afraid.',
      background: 'For three nights wolves have come out of the dark forest and taken the sheep.' },
    user: { name: 'adventurer', persona: 'I am a young adventurer with a sword and a little gold. I want to prove myself.' },
    bot: { name: 'village elder', persona: 'I am the oldest man in the village. I need someone brave to go into the forest and stop the wolves, and I will pay them well.' } },
  { label: 'Forge: the blacksmith', setting: { name: 'The Forge', category: 'Shop',
      description: 'A hot, loud forge full of hammers, horseshoes and half-made swords.',
      background: 'The blacksmith makes weapons for the castle guards.' },
    user: { name: 'squire', persona: 'I am a squire. My knight sent me to fetch his new sword, but I have lost the money.' },
    bot: { name: 'blacksmith', persona: 'I am a strong blacksmith. I am proud of my work and I do not like to be cheated.' } },
  { label: 'Castle gate: the guard', setting: { name: 'Castle Gate', category: 'Castle',
      description: 'A tall iron gate in a stone wall. Torches burn on both sides.',
      background: 'No one may enter the castle at night without the king\'s seal.' },
    user: { name: 'messenger', persona: 'I am a messenger with an urgent letter for the king, but I have no seal.' },
    bot: { name: 'guard', persona: 'I guard the castle gate. I follow the rules, but I am tired and I miss my family.' } },
  { label: 'Tower: the wizard\'s apprentice', setting: { name: 'The Wizard\'s Tower', category: 'Tower',
      description: 'A round room full of books, candles and strange bottles. A black cat sleeps on the table.',
      background: 'The wizard has gone away and left the apprentice in charge.' },
    user: { name: 'farmer', persona: 'I am a farmer. My cow is sick and I hope the wizard can help.' },
    bot: { name: 'apprentice', persona: 'I am the wizard\'s apprentice. I know a few spells, but I am afraid to make mistakes.' } },
  { label: 'Market: the merchant', setting: { name: 'The Market', category: 'Market',
      description: 'A busy market with fruit, cloth and spices. Children run between the stalls.',
      background: 'A merchant from a far land has come to sell rare things.' },
    user: { name: 'cook', persona: 'I am the castle cook. I need spices for the king\'s feast tonight.' },
    bot: { name: 'merchant', persona: 'I am a clever merchant. I sell spices and silk, and I always ask for a high price first.' } },
];

const $ = (id) => document.getElementById(id);
const ui = { log: $('log'), status: $('status'), bar: $('bar'), barWrap: $('barWrap'), form: $('form'), msg: $('msg'), send: $('send'),
  first: $('first'), retry: $('retry'), undo: $('undo'), reset: $('reset'), temp: $('temp'), tempv: $('tempv'),
  sceneName: $('sceneName'), who: $('who'), toggle: $('sceneToggle'), editor: $('sceneEditor'), preset: $('preset'), apply: $('applyScene') };

let wllama = null, ready = false, busy = false, abortCtl = null;
let scene = structuredClone(PRESETS[0]);
let history = [];                          // [{speaker, text, who: 'you'|'npc'}]

// ---------- text helpers ----------
const clean = (s, max = MAX_INPUT) => String(s ?? '')
  .replace(/<\|[^|>]*\|>/g, '')            // no special-token injection
  .replace(/[\u0000-\u001f\u007f]/g, ' ')
  .replace(/\s+/g, ' ').trim().slice(0, max);
const cleanName = (s) => clean(s, 30).replace(/:/g, '') || 'someone';

function header() {
  const s = scene.setting;
  let h = `<|setting|>${clean(s.name, 60)}. ${clean(s.category, 40)}. ${clean(s.description, 300)} ${clean(s.background, 300)}\n`;
  for (const a of [scene.user, scene.bot]) h += `<|persona|>${cleanName(a.name)}: ${clean(a.persona, 300)}\n`;
  return h;
}
function buildPrompt(hist) {
  const head = header(), tail = `<|turn|>${cleanName(scene.bot.name)}:`;
  let budget = PROMPT_CHAR_BUDGET - head.length - tail.length;
  const turns = [];
  for (let i = hist.length - 1; i >= 0; i--) {          // newest turns that fit
    const t = `<|turn|>${cleanName(hist[i].speaker)}: ${hist[i].text}\n`;
    if (t.length > budget) break;
    turns.unshift(t); budget -= t.length;
  }
  return head + turns.join('') + tail;
}
function postprocess(text) {
  return text.replace(/<\|[^|>]*\|>/g, '').split('\n')[0].trim();
}

// ---------- UI ----------
function addMsg(who, name, text) {
  const d = document.createElement('div');
  d.className = `msg ${who}`;
  if (who !== 'sys') { const n = document.createElement('span'); n.className = 'n'; n.textContent = name; d.appendChild(n); }
  const t = document.createElement('span'); t.className = 't'; t.textContent = text; d.appendChild(t);
  ui.log.appendChild(d); ui.log.scrollTop = ui.log.scrollHeight;
  return t;
}
function render() {
  ui.log.textContent = '';
  addMsg('sys', '', `${scene.setting.description}`);
  for (const h of history) addMsg(h.who, h.speaker, h.text);
}
function showScene() {
  ui.sceneName.textContent = scene.setting.name;
  ui.who.textContent = `You: ${scene.user.name} · They: ${scene.bot.name}`;
  ui.msg.placeholder = `Say something to the ${scene.bot.name}…`;
}
function setBusy(b) {
  busy = b;
  const off = !ready || b;
  for (const el of [ui.msg, ui.send, ui.first, ui.reset]) el.disabled = off;
  ui.retry.disabled = off || !history.some((h) => h.who === 'npc');
  ui.undo.disabled = off || history.length === 0;
  ui.send.textContent = b ? 'Stop' : 'Send';
  if (b) ui.send.disabled = false;
}
function status(s) { ui.status.textContent = s; }

function fillEditor() {
  $('sName').value = scene.setting.name; $('sCat').value = scene.setting.category;
  $('sDesc').value = scene.setting.description; $('sBack').value = scene.setting.background;
  $('uName').value = scene.user.name; $('bName').value = scene.bot.name;
  $('uPer').value = scene.user.persona; $('bPer').value = scene.bot.persona;
}
PRESETS.forEach((p, i) => { const o = document.createElement('option'); o.value = i; o.textContent = p.label; ui.preset.appendChild(o); });
ui.preset.addEventListener('change', () => { scene = structuredClone(PRESETS[+ui.preset.value]); fillEditor(); });
ui.toggle.addEventListener('click', () => {
  const open = ui.editor.hidden; ui.editor.hidden = !open; ui.toggle.setAttribute('aria-expanded', String(open));
  if (open) fillEditor();
});
ui.apply.addEventListener('click', () => {
  if (busy) return;
  scene = { setting: { name: clean($('sName').value, 60) || 'Somewhere', category: clean($('sCat').value, 40) || 'Place',
      description: clean($('sDesc').value, 300), background: clean($('sBack').value, 300) },
    user: { name: cleanName($('uName').value), persona: clean($('uPer').value, 300) },
    bot: { name: cleanName($('bName').value), persona: clean($('bPer').value, 300) } };
  history = []; showScene(); render(); ui.editor.hidden = true; ui.toggle.setAttribute('aria-expanded', 'false'); setBusy(false);
});
ui.temp.addEventListener('input', () => { ui.tempv.textContent = Number(ui.temp.value).toFixed(2); });

// ---------- generation ----------
async function generate(temp) {
  const prompt = buildPrompt(history);
  const span = addMsg('npc', scene.bot.name, '…');
  let text = '';
  abortCtl = new AbortController();
  const t0 = performance.now(); let n = 0;
  try {
    await wllama.createCompletion({
      prompt, stream: true, max_tokens: MAX_NEW, abortSignal: abortCtl.signal,
      temperature: temp, top_p: 0.9, top_k: 0, min_p: 0, typical_p: 1,
      repeat_penalty: 1, penalty_repeat: 1, presence_penalty: 0, frequency_penalty: 0,
      stop: ['\n', '<|turn|>', '<|endoftext|>', '<|setting|>', '<|persona|>'],
      seed: Math.floor(Math.random() * 2 ** 31),
      onData: (chunk) => {
        const piece = chunk?.choices?.[0]?.text ?? '';
        if (piece) { text += piece; n++; span.textContent = postprocess(text) || '…'; ui.log.scrollTop = ui.log.scrollHeight; }
      },
    });
  } catch (e) {
    if (e?.name !== 'AbortError' && !String(e?.message || e).toLowerCase().includes('abort')) throw e;
  } finally { abortCtl = null; }
  span.parentElement.remove();
  const dt = (performance.now() - t0) / 1000;
  if (n) status(`${n} tokens in ${dt.toFixed(1)} s`);
  return postprocess(text);
}

async function npcTurn() {
  setBusy(true);
  try {
    const temp = Number(ui.temp.value);
    let text = await generate(temp);
    const lastNpc = [...history].reverse().find((h) => h.who === 'npc');
    if (!text || (lastNpc && text === lastNpc.text)) text = (await generate(Math.min(1.0, temp + 0.3))) || '…';   // same rescue as lchat
    history.push({ speaker: scene.bot.name, text, who: 'npc' });
    addMsg('npc', scene.bot.name, text);
  } catch (e) {
    console.error(e);
    addMsg('sys', '', `Something went wrong: ${String(e?.message || e).slice(0, 160)}`);
  } finally { setBusy(false); ui.msg.focus(); }
}

ui.form.addEventListener('submit', (ev) => {
  ev.preventDefault();
  if (busy) { abortCtl?.abort(); return; }
  const t = clean(ui.msg.value);
  if (!t || !ready) return;
  ui.msg.value = '';
  history.push({ speaker: scene.user.name, text: t, who: 'you' });
  addMsg('you', scene.user.name, t);
  npcTurn();
});
ui.first.addEventListener('click', () => { if (!busy) npcTurn(); });
ui.retry.addEventListener('click', () => {
  if (busy) return;
  const i = history.map((h) => h.who).lastIndexOf('npc');
  if (i < 0) return;
  history = history.slice(0, i); render(); npcTurn();
});
ui.undo.addEventListener('click', () => {
  if (busy || !history.length) return;
  if (history[history.length - 1].who === 'npc') history.pop();
  if (history.length && history[history.length - 1].who === 'you') history.pop();
  render(); setBusy(false);
});
ui.reset.addEventListener('click', () => { if (!busy) { history = []; render(); setBusy(false); } });

// test hook (used by the automated check; harmless for visitors)
window.bytenpc = {
  isReady: () => ready,
  buildPrompt: (sc, hist) => { const keep = scene; scene = sc; try { return buildPrompt(hist); } finally { scene = keep; } },
  complete: (prompt, opts = {}) => wllama.createCompletion({ prompt, max_tokens: MAX_NEW, temperature: 0, top_k: 1, top_p: 1, min_p: 0,
    repeat_penalty: 1, stop: ['\n', '<|turn|>', '<|endoftext|>'], ...opts }),
};

// ---------- load ----------
async function main() {
  showScene(); render(); setBusy(false);
  if (typeof WebAssembly !== 'object') { status('This browser cannot run WebAssembly, so the model cannot load here.'); return; }
  try {
    wllama = new Wllama(WASM, { suppressNativeLog: true });
    ui.barWrap.hidden = false;
    status('Downloading the model (about 50 MB, only the first time)…');
    await wllama.loadModelFromUrl(MODEL_URL, {
      n_ctx: N_CTX, n_batch: 512, n_gpu_layers: 0, warmup: false,
      progressCallback: ({ loaded, total }) => {
        if (total) { const p = Math.round((loaded / total) * 100); ui.bar.style.width = p + '%'; status(`Downloading the model… ${p}%`); }
      },
    });
    ui.barWrap.hidden = true;
    ready = true; setBusy(false);
    status(`Ready. Runs on your device (${wllama.getNumThreads?.() ?? 1} thread${(wllama.getNumThreads?.() ?? 1) > 1 ? 's' : ''}).`);
    ui.msg.focus();
  } catch (e) {
    console.error(e);
    ui.barWrap.hidden = true;
    status(`Could not load the model: ${String(e?.message || e).slice(0, 200)}`);
  }
}
main();
